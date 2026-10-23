//! `mapleocr` command-line tools.

mod bench;
mod config;
mod dataset;
mod hotkey;
mod live;
mod offline;
mod pngout;
mod targets;

use anyhow::{Context, Result, bail};
use clap::{Args, Parser, Subcommand, ValueEnum};
use mapleocr_capture::{BackendKind, Capture, Rect, TargetSelector};

#[derive(Parser)]
#[command(name = "mapleocr", version, about = "mpl-ocr tools (unofficial, not affiliated with Nexon)")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// List capturable windows and monitors.
    Targets(targets::TargetsArgs),
    /// Choose the capture target and save it as the default.
    Select(targets::SelectArgs),
    /// Save full frames and named ROI crops as PNG + JSON sidecars.
    CaptureDataset(dataset::DatasetArgs),
    /// Locate UI anchors and ROIs on the live screen (cuts anchor templates on first run).
    Calibrate(live::CalibrateArgs),
    /// Print readings live as they change.
    Watch(live::WatchArgs),
    /// Print one reading as JSON and exit.
    Read(live::ReadArgs),
    /// Manage your local data packs.
    #[command(subcommand)]
    Pack(live::PackCmd),
    /// Recognize ROIs in saved full-client frames (offline, no game needed).
    Recognize(offline::RecognizeArgs),
    /// Benchmarks.
    #[command(subcommand)]
    Bench(bench::BenchCmd),
}

/// Target selection flags shared by every command that captures. With none
/// given, the saved selection from `mapleocr select` is used, then a
/// "maple" title/process heuristic.
#[derive(Args, Clone, Debug, Default)]
pub struct TargetArgs {
    /// Window handle (decimal or 0x-hex) from `mapleocr targets`.
    #[arg(long, value_parser = parse_hwnd)]
    hwnd: Option<isize>,
    /// Process id.
    #[arg(long)]
    pid: Option<u32>,
    /// Executable name, e.g. msw.exe (case-insensitive).
    #[arg(long)]
    process: Option<String>,
    /// Substring of the window title (case-insensitive).
    #[arg(long)]
    title: Option<String>,
    /// Capture a whole monitor instead of a window (index from `mapleocr targets`).
    #[arg(long)]
    monitor: Option<usize>,
    /// Capture backend.
    #[arg(long, value_enum)]
    backend: Option<BackendArg>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
pub enum BackendArg {
    Auto,
    Wgc,
    Dxgi,
}

impl From<BackendArg> for BackendKind {
    fn from(b: BackendArg) -> Self {
        match b {
            BackendArg::Auto => Self::Auto,
            BackendArg::Wgc => Self::Wgc,
            BackendArg::Dxgi => Self::Dxgi,
        }
    }
}

fn parse_hwnd(s: &str) -> Result<isize, String> {
    let r = match s.strip_prefix("0x").or_else(|| s.strip_prefix("0X")) {
        Some(hex) => isize::from_str_radix(hex, 16),
        None => s.parse(),
    };
    r.map_err(|e| e.to_string())
}

impl TargetArgs {
    pub fn selector(&self) -> TargetSelector {
        TargetSelector {
            hwnd: self.hwnd,
            pid: self.pid,
            process: self.process.clone(),
            title: self.title.clone(),
            monitor: self.monitor,
        }
    }

    /// Flags, else the saved config, else the heuristic (empty selector).
    pub fn open(&self) -> Result<Capture> {
        let cfg = config::Config::load()?;
        let mut selector = self.selector();
        if selector.is_empty() {
            selector = cfg.target.clone().unwrap_or_default();
        }
        let kind = self.backend.map(Into::into).unwrap_or(cfg.backend);
        let cap = Capture::open(selector.clone(), kind).with_context(|| {
            if selector.is_empty() {
                "no window looks like MapleStory; run `mapleocr targets` and then `mapleocr select`".to_string()
            } else {
                format!("could not open capture for {selector:?}")
            }
        })?;
        eprintln!("capturing {} via {}", cap.target().describe(), cap.backend_name());
        Ok(cap)
    }
}

/// `name=x,y,w,h` in client pixels.
#[derive(Clone, Debug)]
pub struct NamedRoi {
    pub name: String,
    pub rect: Rect,
}

impl std::str::FromStr for NamedRoi {
    type Err = anyhow::Error;

    fn from_str(s: &str) -> Result<Self> {
        let (name, coords) = s.split_once('=').context("expected name=x,y,w,h")?;
        let v: Vec<i64> = coords.split(',').map(|p| p.trim().parse()).collect::<Result<_, _>>()?;
        let [x, y, w, h] = v[..] else { bail!("expected 4 numbers in {s:?}") };
        if w <= 0 || h <= 0 {
            bail!("ROI {name} has empty size");
        }
        if name.is_empty() || !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            bail!("ROI name {name:?} must be [A-Za-z0-9_-]+");
        }
        Ok(Self { name: name.to_owned(), rect: Rect::new(x as i32, y as i32, w as u32, h as u32) })
    }
}

fn main() -> Result<()> {
    mapleocr_capture::init_dpi_awareness();
    match Cli::parse().cmd {
        Cmd::Targets(a) => targets::run_targets(a),
        Cmd::Select(a) => targets::run_select(a),
        Cmd::CaptureDataset(a) => dataset::run(a),
        Cmd::Calibrate(a) => live::calibrate_cmd(a),
        Cmd::Watch(a) => live::watch(a),
        Cmd::Read(a) => live::read(a),
        Cmd::Pack(c) => live::pack(c),
        Cmd::Recognize(a) => offline::recognize(a),
        Cmd::Bench(c) => bench::run(c),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_named_roi() {
        let r: NamedRoi = "map_name=10,-2,300,24".parse().unwrap();
        assert_eq!(r.name, "map_name");
        assert_eq!(r.rect, Rect::new(10, -2, 300, 24));
        assert!("x=1,2,3".parse::<NamedRoi>().is_err());
        assert!("x=1,2,0,4".parse::<NamedRoi>().is_err());
        assert!("bad name=1,2,3,4".parse::<NamedRoi>().is_err());
    }

    #[test]
    fn parses_hwnd_hex_and_decimal() {
        assert_eq!(parse_hwnd("0x30532"), Ok(0x30532));
        assert_eq!(parse_hwnd("198962"), Ok(198962));
        assert!(parse_hwnd("zz").is_err());
    }
}
