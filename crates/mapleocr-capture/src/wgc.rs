//! Windows Graphics Capture backend (windowed and borderless fullscreen).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::Duration;

use windows::Foundation::TypedEventHandler;
use windows::Graphics::Capture::{
    Direct3D11CaptureFrame, Direct3D11CaptureFramePool, GraphicsCaptureItem, GraphicsCaptureSession,
};
use windows::Graphics::DirectX::Direct3D11::IDirect3DDevice;
use windows::Graphics::DirectX::DirectXPixelFormat;
use windows::Graphics::SizeInt32;
use windows::Win32::Foundation::{CloseHandle, HANDLE, WAIT_OBJECT_0};
use windows::Win32::Graphics::Direct3D11::ID3D11Texture2D;
use windows::Win32::Graphics::Dxgi::IDXGIDevice;
use windows::Win32::System::Threading::{CreateEventW, SetEvent, WaitForSingleObject};
use windows::Win32::System::WinRT::Direct3D11::{CreateDirect3D11DeviceFromDXGIDevice, IDirect3DDxgiInterfaceAccess};
use windows::Win32::System::WinRT::Graphics::Capture::IGraphicsCaptureItemInterop;
use windows::Win32::System::WinRT::{RO_INIT_MULTITHREADED, RoInitialize};
use windows::core::{IInspectable, Interface, factory};

use crate::d3d::{RoiStager, create_device};
use crate::target::{CaptureTarget, WindowGeometry};
use crate::time::{Ticks, now_ticks};
use crate::{CaptureBackend, CaptureError, Frame, Rect, Result};

const PIXEL_FORMAT: DirectXPixelFormat = DirectXPixelFormat::B8G8R8A8UIntNormalized;
const POOL_BUFFERS: i32 = 2;

/// Auto-reset Win32 event, shared with WGC callback threads.
struct Event(HANDLE);

// SAFETY: event handles may be signalled and waited on from any thread.
unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    fn new() -> Result<Self> {
        // SAFETY: anonymous auto-reset event, initially unsignalled.
        Ok(Self(unsafe { CreateEventW(None, false, false, None)? }))
    }

    fn signal(&self) {
        // SAFETY: handle is open for as long as self lives.
        let _ = unsafe { SetEvent(self.0) };
    }

    /// True if signalled before the timeout.
    fn wait(&self, timeout: Duration) -> bool {
        let ms = timeout.as_millis().min(u32::MAX as u128 - 1) as u32;
        // SAFETY: handle is open for as long as self lives.
        unsafe { WaitForSingleObject(self.0, ms) == WAIT_OBJECT_0 }
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        // SAFETY: we own the handle.
        let _ = unsafe { CloseHandle(self.0) };
    }
}

#[derive(Clone, Copy)]
struct FrameMeta {
    frame_id: u64,
    present: Ticks,
    acquired: Ticks,
    ready: Ticks,
    client_size: (u32, u32),
}

pub struct WgcBackend {
    target: CaptureTarget,
    winrt_device: IDirect3DDevice,
    pool: Direct3D11CaptureFramePool,
    session: GraphicsCaptureSession,
    frame_ready: Arc<Event>,
    closed: Arc<AtomicBool>,
    pool_size: SizeInt32,
    origin: (i32, i32),
    client_size: (u32, u32),
    stager: RoiStager,
    next_id: u64,
    last: Option<FrameMeta>,
}

impl WgcBackend {
    pub fn open(target: &CaptureTarget) -> Result<Self> {
        // SAFETY: idempotent per thread; an existing STA/MTA init is fine too.
        let _ = unsafe { RoInitialize(RO_INIT_MULTITHREADED) };

        let (device, context) = create_device(None)?;
        let dxgi: IDXGIDevice = device.cast()?;
        // SAFETY: dxgi is a live DXGI device.
        let inspectable: IInspectable = unsafe { CreateDirect3D11DeviceFromDXGIDevice(&dxgi)? };
        let winrt_device: IDirect3DDevice = inspectable.cast()?;

        let interop = factory::<GraphicsCaptureItem, IGraphicsCaptureItemInterop>()?;
        // SAFETY: handles come from enumeration; stale ones make these calls fail.
        let item: GraphicsCaptureItem = unsafe {
            match target {
                CaptureTarget::Window(w) => interop.CreateForWindow(w.hwnd())?,
                CaptureTarget::Monitor(m) => interop.CreateForMonitor(m.hmonitor())?,
            }
        };
        let size = item.Size()?;
        let pool = Direct3D11CaptureFramePool::CreateFreeThreaded(&winrt_device, PIXEL_FORMAT, POOL_BUFFERS, size)?;

        let frame_ready = Arc::new(Event::new()?);
        let closed = Arc::new(AtomicBool::new(false));
        {
            let ev = frame_ready.clone();
            pool.FrameArrived(&TypedEventHandler::new(move |_, _| {
                ev.signal();
                Ok(())
            }))?;
        }
        {
            let ev = frame_ready.clone();
            let closed = closed.clone();
            item.Closed(&TypedEventHandler::<GraphicsCaptureItem, IInspectable>::new(move |_, _| {
                closed.store(true, Ordering::Release);
                ev.signal();
                Ok(())
            }))?;
        }

        let session = pool.CreateCaptureSession(&item)?;
        // Keep the cursor out of ROIs (it would defeat dirty detection) and drop
        // the yellow border where the OS allows it. Both are best-effort.
        let _ = session.SetIsCursorCaptureEnabled(false);
        let _ = session.SetIsBorderRequired(false);
        session.StartCapture()?;

        let mut me = Self {
            target: target.clone(),
            winrt_device,
            pool,
            session,
            frame_ready,
            closed,
            pool_size: size,
            origin: (0, 0),
            client_size: (size.Width as u32, size.Height as u32),
            stager: RoiStager::new(device, context),
            next_id: 1,
            last: None,
        };
        me.refresh_geometry()?;
        Ok(me)
    }

    fn refresh_geometry(&mut self) -> Result<()> {
        match &self.target {
            CaptureTarget::Window(w) => {
                let g = WindowGeometry::query(w.hwnd())?;
                self.origin = g.client_offset_in_frame();
                self.client_size = g.client_size;
            }
            CaptureTarget::Monitor(m) => {
                self.origin = (0, 0);
                self.client_size = m.size();
            }
        }
        Ok(())
    }

    /// Take the newest queued frame, releasing older ones back to the pool.
    fn latest_frame(&self) -> Option<Direct3D11CaptureFrame> {
        let mut newest = None;
        while let Ok(f) = self.pool.TryGetNextFrame() {
            if let Some(old) = newest.replace(f) {
                let _ = Direct3D11CaptureFrame::Close(&old);
            }
        }
        newest
    }

    fn meta_to_frame(&self, m: FrameMeta) -> Frame<'_> {
        Frame {
            frame_id: m.frame_id,
            present: m.present,
            acquired: m.acquired,
            ready: m.ready,
            client_size: m.client_size,
            rois: self.stager.images(),
        }
    }
}

impl CaptureBackend for WgcBackend {
    fn name(&self) -> &'static str {
        "wgc"
    }

    fn set_rois(&mut self, rois: &[Rect]) -> Result<()> {
        self.last = None;
        self.stager.set_rois(rois)
    }

    fn set_active(&mut self, active: &[bool]) {
        self.stager.set_active(active);
    }

    fn next_frame(&mut self, timeout: Duration) -> Result<Option<Frame<'_>>> {
        if self.closed.load(Ordering::Acquire) {
            return Err(CaptureError::TargetLost);
        }
        if !self.frame_ready.wait(timeout) {
            return Ok(None);
        }
        if self.closed.load(Ordering::Acquire) {
            return Err(CaptureError::TargetLost);
        }
        let acquired = now_ticks();
        let Some(frame) = self.latest_frame() else { return Ok(None) };

        let content = frame.ContentSize()?;
        if content != self.pool_size {
            self.pool.Recreate(&self.winrt_device, PIXEL_FORMAT, POOL_BUFFERS, content)?;
            self.pool_size = content;
            self.refresh_geometry()?;
        }

        let present = Ticks(frame.SystemRelativeTime()?.Duration);
        let access: IDirect3DDxgiInterfaceAccess = frame.Surface()?.cast()?;
        // SAFETY: WGC surfaces are always backed by an ID3D11Texture2D.
        let tex: ID3D11Texture2D = unsafe { access.GetInterface()? };
        self.stager.stage(&tex, self.origin)?;
        drop(tex);
        let _ = frame.Close();
        let ready = now_ticks();

        let meta = FrameMeta { frame_id: self.next_id, present, acquired, ready, client_size: self.client_size };
        self.next_id += 1;
        self.last = Some(meta);
        Ok(Some(self.meta_to_frame(meta)))
    }

    fn last_frame(&self) -> Option<Frame<'_>> {
        self.last.map(|m| self.meta_to_frame(m))
    }

    fn client_size(&self) -> (u32, u32) {
        self.client_size
    }
}

impl Drop for WgcBackend {
    fn drop(&mut self) {
        let _ = self.session.Close();
        let _ = self.pool.Close();
    }
}
