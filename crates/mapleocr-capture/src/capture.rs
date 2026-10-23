//! Backend selection, fullscreen fallback, and rebinding to a recreated window.

use std::time::{Duration, Instant};

use crate::dxgi::DxgiBackend;
use crate::target::{CaptureTarget, TargetSelector, WindowGeometry};
use crate::wgc::WgcBackend;
use crate::{CaptureBackend, CaptureError, Frame, Rect, Result};

/// WGC always delivers a first frame right after `StartCapture`. If none has
/// arrived after this long, the window is most likely in exclusive fullscreen.
const FIRST_FRAME_GRACE: Duration = Duration::from_millis(1500);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum BackendKind {
    /// WGC, switching to DXGI duplication for exclusive fullscreen windows.
    #[default]
    Auto,
    Wgc,
    Dxgi,
}

/// A capture session bound to a [`TargetSelector`] rather than to one HWND:
/// if the game window closes and reopens, the next matching window is picked
/// up again with the same ROIs.
pub struct Capture {
    selector: TargetSelector,
    kind: BackendKind,
    target: CaptureTarget,
    backend: Box<dyn CaptureBackend>,
    rois: Vec<Rect>,
    lost: bool,
    opened_at: Instant,
    frames_since_open: u64,
    fallback_checked: bool,
}

fn open_backend(target: &CaptureTarget, kind: BackendKind) -> Result<Box<dyn CaptureBackend>> {
    Ok(match kind {
        BackendKind::Wgc => Box::new(WgcBackend::open(target)?),
        BackendKind::Dxgi => Box::new(DxgiBackend::open(target)?),
        BackendKind::Auto => match WgcBackend::open(target) {
            Ok(b) => Box::new(b),
            Err(e) => match target {
                CaptureTarget::Window(w) if WindowGeometry::query(w.hwnd())?.covers_monitor() => {
                    Box::new(DxgiBackend::open(target)?)
                }
                _ => return Err(e),
            },
        },
    })
}

impl Capture {
    /// Opens capture and makes the calling thread per-monitor DPI aware;
    /// use the returned `Capture` from this thread.
    pub fn open(selector: TargetSelector, kind: BackendKind) -> Result<Self> {
        crate::target::init_thread_dpi_awareness();
        let target = selector.resolve()?;
        let backend = open_backend(&target, kind)?;
        Ok(Self {
            selector,
            kind,
            target,
            backend,
            rois: Vec::new(),
            lost: false,
            opened_at: Instant::now(),
            frames_since_open: 0,
            fallback_checked: false,
        })
    }

    pub fn target(&self) -> &CaptureTarget {
        &self.target
    }

    pub fn backend_name(&self) -> &'static str {
        self.backend.name()
    }

    pub fn client_size(&self) -> (u32, u32) {
        self.backend.client_size()
    }

    pub fn set_rois(&mut self, rois: &[Rect]) -> Result<()> {
        self.rois = rois.to_vec();
        self.backend.set_rois(rois)
    }

    /// Which ROIs to copy off the GPU for the next frames. Inactive ROIs keep
    /// their last pixels; use this to skip windows that are closed or ROIs
    /// that are not due yet.
    pub fn set_active(&mut self, active: &[bool]) {
        self.backend.set_active(active);
    }

    pub fn last_frame(&self) -> Option<Frame<'_>> {
        self.backend.last_frame()
    }

    fn reopen(&mut self, target: CaptureTarget, kind: BackendKind) -> Result<()> {
        let mut backend = open_backend(&target, kind)?;
        backend.set_rois(&self.rois)?;
        self.backend = backend;
        self.target = target;
        self.opened_at = Instant::now();
        self.frames_since_open = 0;
        self.fallback_checked = false;
        Ok(())
    }

    fn maybe_fall_back_to_dxgi(&mut self) -> Result<()> {
        if self.kind != BackendKind::Auto
            || self.fallback_checked
            || self.frames_since_open > 0
            || self.backend.name() != "wgc"
            || self.opened_at.elapsed() < FIRST_FRAME_GRACE
        {
            return Ok(());
        }
        self.fallback_checked = true;
        let CaptureTarget::Window(w) = &self.target else { return Ok(()) };
        let g = WindowGeometry::query(w.hwnd())?;
        if !g.minimized && g.covers_monitor() {
            self.reopen(self.target.clone(), BackendKind::Dxgi)?;
        }
        Ok(())
    }

    /// Wait up to `timeout` for the next frame. Returns `Ok(None)` when nothing
    /// changed, the target is minimized, or it is gone and not yet back.
    pub fn next_frame(&mut self, timeout: Duration) -> Result<Option<Frame<'_>>> {
        if self.lost {
            match self.selector.resolve() {
                Ok(t) => {
                    self.reopen(t, self.kind)?;
                    self.lost = false;
                }
                Err(_) => {
                    // No OS event tells us the window is back, so back off.
                    std::thread::sleep(timeout);
                    return Ok(None);
                }
            }
        }
        match self.maybe_fall_back_to_dxgi() {
            Err(CaptureError::TargetLost) => {
                self.lost = true;
                return Ok(None);
            }
            r => r?,
        }
        match self.backend.next_frame(timeout) {
            Ok(Some(f)) => {
                self.frames_since_open += 1;
                Ok(Some(f))
            }
            Ok(None) => Ok(None),
            Err(CaptureError::TargetLost) => {
                self.lost = true;
                Ok(None)
            }
            Err(e) => Err(e),
        }
    }
}
