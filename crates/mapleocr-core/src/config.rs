//! User configuration and data locations (`%LOCALAPPDATA%\mapleocr`).
//!
//! ```text
//! config.toml          saved capture target + backend
//! anchors/             anchor templates cut from the user's screen
//! packs/maps.json      map-name data pack (user-built)
//! layout.yaml          optional override of the built-in layout
//! ```

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::{Error, Result};

#[cfg(windows)]
pub use mapleocr_capture::{BackendKind, TargetSelector};

pub fn data_dir() -> Result<PathBuf> {
    dirs::data_local_dir()
        .map(|d| d.join("mapleocr"))
        .ok_or_else(|| Error::Config("no local app data directory".into()))
}

#[cfg(windows)]
#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub target: Option<TargetSelector>,
    pub backend: BackendKind,
}

#[cfg(not(windows))]
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct Config {}

impl Config {
    fn path() -> Result<PathBuf> {
        Ok(data_dir()?.join("config.toml"))
    }

    pub fn load() -> Result<Self> {
        let p = Self::path()?;
        match std::fs::read_to_string(&p) {
            Ok(s) => toml::from_str(&s).map_err(|e| Error::Config(format!("{}: {e}", p.display()))),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e.into()),
        }
    }

    pub fn save(&self) -> Result<PathBuf> {
        let p = Self::path()?;
        std::fs::create_dir_all(p.parent().expect("config path has a parent"))?;
        std::fs::write(&p, toml::to_string_pretty(self).map_err(|e| Error::Config(e.to_string()))?)?;
        Ok(p)
    }
}
