//! Screen capture for mpl-ocr.
//!
//! Pixels only: frames come from Windows Graphics Capture (windowed and
//! borderless) or DXGI Desktop Duplication (exclusive fullscreen). Only the
//! requested ROI rectangles are copied off the GPU, into per-ROI staging
//! textures and pre-allocated CPU buffers, so the hot path never touches the
//! full frame and never allocates.
//!
//! All rectangles are in *client-area pixels* of the target window (or monitor
//! pixels for a monitor target). Backends translate them into texture space.

#![cfg(windows)]

mod capture;
pub mod d3d;
mod dxgi;
mod target;
mod time;
mod wgc;

pub use capture::{BackendKind, Capture};
pub use d3d::RoiStager;
pub use dxgi::DxgiBackend;
pub use target::{
    CaptureTarget, MonitorInfo, TargetSelector, WindowGeometry, WindowInfo, init_dpi_awareness,
    init_thread_dpi_awareness, list_monitors, list_windows,
};
pub use time::{Ticks, now_ticks};
pub use wgc::WgcBackend;

use std::time::Duration;

use mapleocr_imgproc::BgraView;
/// Rectangles handed to capture are in client-area pixels.
pub use mapleocr_imgproc::Rect;

/// CPU copy of one ROI, tightly packed BGRA8 (`width * 4` bytes per row).
///
/// Pixels of the ROI that fell outside the captured texture are zero.
#[derive(Debug)]
pub struct RoiImage {
    rect: Rect,
    data: Vec<u8>,
}

impl RoiImage {
    pub(crate) fn new(rect: Rect) -> Self {
        Self { rect, data: vec![0; rect.w as usize * rect.h as usize * 4] }
    }

    pub fn rect(&self) -> Rect {
        self.rect
    }

    pub fn width(&self) -> u32 {
        self.rect.w
    }

    pub fn height(&self) -> u32 {
        self.rect.h
    }

    /// BGRA8 pixels, row-major, stride `width * 4`.
    pub fn data(&self) -> &[u8] {
        &self.data
    }

    pub fn view(&self) -> BgraView<'_> {
        BgraView::packed(self.rect.w, self.rect.h, &self.data)
    }

    pub(crate) fn data_mut(&mut self) -> &mut [u8] {
        &mut self.data
    }
}

/// A captured frame: metadata plus the ROI crops, borrowed from the backend.
#[derive(Clone, Copy, Debug)]
pub struct Frame<'a> {
    /// Monotonic per-backend frame counter, starting at 1.
    pub frame_id: u64,
    /// When the compositor or swapchain produced the frame.
    pub present: Ticks,
    /// When we woke up holding the frame.
    pub acquired: Ticks,
    /// When every ROI was readable in CPU memory.
    pub ready: Ticks,
    /// Client-area size of the target at capture time.
    pub client_size: (u32, u32),
    pub rois: &'a [RoiImage],
}

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("capture target is gone (window closed or monitor removed)")]
    TargetLost,
    #[error("no capture target matched {0}")]
    NoMatch(String),
    #[error("capture is unsupported for this target: {0}")]
    Unsupported(&'static str),
    #[error(transparent)]
    Windows(#[from] windows::core::Error),
}

pub type Result<T, E = CaptureError> = std::result::Result<T, E>;

/// A source of frames. Implementations block on an OS event; they never poll.
pub trait CaptureBackend {
    fn name(&self) -> &'static str;

    /// Replace the ROI set. Allocates staging resources; call it outside the
    /// hot path (at calibration or after a resize).
    fn set_rois(&mut self, rois: &[Rect]) -> Result<()>;

    /// Wait up to `timeout` for a new frame. `Ok(None)` means nothing changed
    /// on screen in that time (or the target is minimized).
    fn next_frame(&mut self, timeout: Duration) -> Result<Option<Frame<'_>>>;

    /// The most recently delivered frame, if any.
    fn last_frame(&self) -> Option<Frame<'_>>;

    /// Current client-area size of the target.
    fn client_size(&self) -> (u32, u32);
}
