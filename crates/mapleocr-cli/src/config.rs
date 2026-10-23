//! `%LOCALAPPDATA%\mapleocr\config.toml`.

use std::path::PathBuf;

use anyhow::{Context, Result};
use mapleocr_capture::{BackendKind, TargetSelector};
use serde::{Deserialize, Serialize};

#[derive(Debug, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub target: Option<TargetSelector>,
    pub backend: BackendKind,
}

pub fn data_dir() -> Result<PathBuf> {
    Ok(dirs::data_local_dir().context("no local app data directory")?.join("mapleocr"))
}

fn path() -> Result<PathBuf> {
    Ok(data_dir()?.join("config.toml"))
}

impl Config {
    pub fn load() -> Result<Self> {
        let p = path()?;
        match std::fs::read_to_string(&p) {
            Ok(s) => toml::from_str(&s).with_context(|| format!("parsing {}", p.display())),
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(Self::default()),
            Err(e) => Err(e).with_context(|| format!("reading {}", p.display())),
        }
    }

    pub fn save(&self) -> Result<PathBuf> {
        let p = path()?;
        std::fs::create_dir_all(p.parent().expect("config path has a parent"))?;
        std::fs::write(&p, toml::to_string_pretty(self)?)?;
        Ok(p)
    }
}
