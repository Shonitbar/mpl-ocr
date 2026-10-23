//! Recognizer tiers behind a common [`Recognizer`] trait, plus the
//! [`Reading`] type every tier's output ends up in.
//!
//! M1 ships the fallback tier ([`WinOcr`], the OCR engine built into
//! Windows). Glyph, sprite-digit, icon and CRNN tiers arrive in M2-M4.

pub mod glyph;
pub mod preprocess;
#[cfg(windows)]
mod winocr;

use std::sync::Arc;

use serde::Serialize;

pub use glyph::Atlas;
pub use mapleocr_semantic::MapName;
#[cfg(windows)]
pub use winocr::WinOcr;

#[derive(Debug, thiserror::Error)]
pub enum RecogError {
    #[error("recognizer unavailable: {0}")]
    Unavailable(String),
    #[cfg(windows)]
    #[error(transparent)]
    Windows(#[from] windows::core::Error),
}

pub type Result<T, E = RecogError> = std::result::Result<T, E>;

/// Raw output of a recognizer, before semantic parsing.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Recognition {
    /// Text lines, top to bottom.
    pub lines: Vec<String>,
    /// Engine-reported confidence, if the engine has one.
    pub confidence: Option<f32>,
}

/// One recognizer tier. Implementations may hold per-thread state (e.g. a
/// WinRT engine), so they are created and used on a single worker thread.
pub trait Recognizer {
    fn engine(&self) -> &'static str;
    /// `img` is already color-keyed and cropped: dark text on a light background.
    fn recognize(&mut self, img: mapleocr_imgproc::BgraView<'_>) -> Result<Recognition>;
}

/// Typed reading values.
#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Value {
    Text {
        text: String,
    },
    Map(MapName),
    Number {
        value: i64,
    },
    /// `current/max`, e.g. HP and MP.
    Ratio {
        current: i64,
        max: i64,
    },
    /// EXP points and percent of the current level.
    Exp {
        value: i64,
        percent: f32,
    },
    /// The ROI is visible but holds no text (e.g. a collapsed minimap).
    Empty,
}

/// A recognized value for one ROI, with provenance and timing.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Reading {
    pub roi: Arc<str>,
    pub value: Value,
    /// 0..=1. Below 0.6 means "unverified": no data pack confirmed it.
    pub confidence: f32,
    /// Which tier produced it: `winocr`, `glyph`, `cache`, ...
    pub engine: &'static str,
    /// Recognition time in milliseconds (0 for cache hits).
    pub ms: f32,
    /// Capture frame the pixels came from.
    pub frame_id: u64,
    /// QPC timestamp (100 ns) when the frame was acquired.
    pub captured_at: i64,
    /// QPC timestamp (100 ns) when this reading was published.
    pub published_at: i64,
    /// Raw recognizer text, newline-joined, for debugging.
    pub raw: String,
    /// True if the value was snapped to a data-pack entry.
    pub snapped: bool,
}

impl Reading {
    /// Capture-to-publish latency in milliseconds.
    pub fn latency_ms(&self) -> f32 {
        (self.published_at - self.captured_at) as f32 / 10_000.0
    }
}
