//! mpl-ocr runtime.
//!
//! ```no_run
//! let ocr = mapleocr_core::MapleOcr::auto()?;          // find window + calibrate
//! ocr.subscribe(&["map_name"], |r| println!("{:?}", r.value))?;
//! if let Some(r) = ocr.read("map_name")? { println!("{:?} in {:.1} ms", r.value, r.ms) }
//! # Ok::<(), mapleocr_core::Error>(())
//! ```
//!
//! Threads: a capture thread waits on frames, hashes each ROI and sends only
//! changed, settled ROIs to a recognition worker. The worker checks a
//! hash-keyed cache, runs the recognizer tier, snaps against data packs, and
//! publishes readings. Latest readings are read lock-free.

pub mod config;
mod pipeline;
#[cfg(windows)]
mod runtime;

pub use config::{Config, data_dir};
pub use pipeline::{Packs, Pipeline};
#[cfg(windows)]
pub use runtime::{MapleOcr, Options, Status, SubscriptionId};

pub use mapleocr_layout::{AnchorTemplates, Calibration, Layout};
pub use mapleocr_recog::{Reading, Value};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Layout(#[from] mapleocr_layout::LayoutError),
    #[error(transparent)]
    Recog(#[from] mapleocr_recog::RecogError),
    #[error(transparent)]
    Pack(#[from] mapleocr_semantic::PackError),
    #[cfg(windows)]
    #[error(transparent)]
    Capture(#[from] mapleocr_capture::CaptureError),
    #[error("config: {0}")]
    Config(String),
    #[error("unknown ROI {0:?}")]
    UnknownRoi(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
}

pub type Result<T, E = Error> = std::result::Result<T, E>;
