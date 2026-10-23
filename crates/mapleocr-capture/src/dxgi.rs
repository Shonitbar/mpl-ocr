//! DXGI Desktop Duplication backend, for exclusive fullscreen where WGC
//! window capture delivers nothing. Captures the game's monitor and crops to
//! the window's client area.

use std::time::Duration;

use windows::Win32::Graphics::Direct3D11::{ID3D11Device, ID3D11Texture2D};
use windows::Win32::Graphics::Dxgi::{
    CreateDXGIFactory1, DXGI_ERROR_ACCESS_LOST, DXGI_ERROR_WAIT_TIMEOUT, DXGI_OUTDUPL_FRAME_INFO, IDXGIAdapter,
    IDXGIFactory1, IDXGIOutput1, IDXGIOutputDuplication, IDXGIResource,
};
use windows::Win32::Graphics::Gdi::HMONITOR;
use windows::core::Interface;

use crate::d3d::{RoiStager, create_device};
use crate::target::{CaptureTarget, WindowGeometry};
use crate::time::{Ticks, now_ticks, ticks_from_qpc};
use crate::{CaptureBackend, CaptureError, Frame, Rect, Result};

#[derive(Clone, Copy)]
struct FrameMeta {
    frame_id: u64,
    present: Ticks,
    acquired: Ticks,
    ready: Ticks,
    client_size: (u32, u32),
}

pub struct DxgiBackend {
    target: CaptureTarget,
    device: ID3D11Device,
    output: IDXGIOutput1,
    /// None after ACCESS_LOST until re-duplication succeeds.
    dup: Option<IDXGIOutputDuplication>,
    output_origin: (i32, i32),
    client_size: (u32, u32),
    stager: RoiStager,
    next_id: u64,
    last: Option<FrameMeta>,
}

fn find_output(hmon: HMONITOR) -> Result<(IDXGIAdapter, IDXGIOutput1, (i32, i32))> {
    // SAFETY: COM enumeration; loops end at the first DXGI_ERROR_NOT_FOUND.
    unsafe {
        let factory: IDXGIFactory1 = CreateDXGIFactory1()?;
        let mut a = 0;
        while let Ok(adapter) = factory.EnumAdapters1(a) {
            let mut o = 0;
            while let Ok(output) = adapter.EnumOutputs(o) {
                let desc = output.GetDesc()?;
                if desc.Monitor == hmon {
                    let r = desc.DesktopCoordinates;
                    return Ok((adapter.cast()?, output.cast()?, (r.left, r.top)));
                }
                o += 1;
            }
            a += 1;
        }
    }
    Err(CaptureError::NoMatch("DXGI output for the target's monitor".into()))
}

impl DxgiBackend {
    pub fn open(target: &CaptureTarget) -> Result<Self> {
        let (hmon, client_size) = match target {
            CaptureTarget::Window(w) => {
                let g = WindowGeometry::query(w.hwnd())?;
                (g.monitor, g.client_size)
            }
            CaptureTarget::Monitor(m) => (m.hmonitor(), m.size()),
        };
        let (adapter, output, output_origin) = find_output(hmon)?;
        let (device, context) = create_device(Some(&adapter))?;
        // SAFETY: device lives on the output's adapter, as DuplicateOutput requires.
        let dup = unsafe { output.DuplicateOutput(&device)? };
        Ok(Self {
            target: target.clone(),
            device: device.clone(),
            output,
            dup: Some(dup),
            output_origin,
            client_size,
            stager: RoiStager::new(device, context),
            next_id: 1,
            last: None,
        })
    }

    /// Client origin in output-texture pixels.
    fn client_origin(&mut self) -> Result<(i32, i32)> {
        match &self.target {
            CaptureTarget::Window(w) => {
                let g = WindowGeometry::query(w.hwnd())?;
                self.client_size = g.client_size;
                Ok((g.client_origin.0 - self.output_origin.0, g.client_origin.1 - self.output_origin.1))
            }
            CaptureTarget::Monitor(_) => Ok((0, 0)),
        }
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

impl CaptureBackend for DxgiBackend {
    fn name(&self) -> &'static str {
        "dxgi"
    }

    fn set_rois(&mut self, rois: &[Rect]) -> Result<()> {
        self.last = None;
        self.stager.set_rois(rois)
    }

    fn next_frame(&mut self, timeout: Duration) -> Result<Option<Frame<'_>>> {
        if self.dup.is_none() {
            // Mode switch or secure desktop; retry duplication. There is no
            // event to wait on here, so back off for the timeout on failure.
            // SAFETY: same device/output pairing as in `open`.
            match unsafe { self.output.DuplicateOutput(&self.device) } {
                Ok(d) => self.dup = Some(d),
                Err(_) => {
                    std::thread::sleep(timeout);
                    return Ok(None);
                }
            }
        }
        let dup = self.dup.as_ref().expect("duplication");
        let ms = timeout.as_millis().min(u32::MAX as u128 - 1) as u32;
        let mut info = DXGI_OUTDUPL_FRAME_INFO::default();
        let mut resource: Option<IDXGIResource> = None;
        // SAFETY: valid out-pointers; every successful acquire is released below.
        match unsafe { dup.AcquireNextFrame(ms, &mut info, &mut resource) } {
            Ok(()) => {}
            Err(e) if e.code() == DXGI_ERROR_WAIT_TIMEOUT => return Ok(None),
            Err(e) if e.code() == DXGI_ERROR_ACCESS_LOST => {
                self.dup = None;
                return Ok(None);
            }
            Err(e) => return Err(e.into()),
        }
        let acquired = now_ticks();

        let staged = (|| -> Result<Option<Ticks>> {
            // LastPresentTime == 0 means only the mouse moved.
            if info.LastPresentTime == 0 {
                return Ok(None);
            }
            let tex: ID3D11Texture2D = resource.as_ref().expect("desktop resource").cast()?;
            let origin = self.client_origin()?;
            self.stager.stage(&tex, origin)?;
            Ok(Some(ticks_from_qpc(info.LastPresentTime)))
        })();
        drop(resource);
        if let Some(dup) = &self.dup {
            // SAFETY: pairs with the successful AcquireNextFrame above.
            let _ = unsafe { dup.ReleaseFrame() };
        }
        let Some(present) = staged? else { return Ok(None) };
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
