//! `mapleocr capture-dataset`: save frames on a hotkey or interval.
//!
//! Output (default `fixtures/local/session-<unix>/`, git-ignored):
//! `0001_full.png`, `0001_<roi>.png`, `0001.json`. Captures stay on the
//! user's machine; they contain game pixels and must not be committed.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

use anyhow::{Context, Result};
use clap::Args;
use mapleocr_capture::{CaptureTarget, Frame, Rect};

use crate::hotkey::Hotkey;
use crate::pngout::write_png;
use crate::{NamedRoi, TargetArgs};

#[derive(Args)]
pub struct DatasetArgs {
    #[command(flatten)]
    target: TargetArgs,
    /// Output directory. Default: fixtures/local/session-<unix time>.
    #[arg(long)]
    out: Option<PathBuf>,
    /// Global hotkey that saves the current frame.
    #[arg(long, default_value = "F9")]
    hotkey: String,
    /// Also save automatically every N milliseconds.
    #[arg(long)]
    interval_ms: Option<u64>,
    /// Stop after this many saves.
    #[arg(long)]
    count: Option<u32>,
    /// Named crop to save alongside the full frame: name=x,y,w,h (client pixels). Repeatable.
    #[arg(long = "roi")]
    rois: Vec<NamedRoi>,
    /// Free-text label stored in every sidecar (e.g. "inventory open, 1366x768").
    #[arg(long)]
    label: Option<String>,
    /// Do not save the full client frame, only the named ROIs.
    #[arg(long)]
    no_full: bool,
}

fn unix_ms() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_millis()).unwrap_or(0)
}

fn rects(full: Option<(u32, u32)>, named: &[NamedRoi]) -> Vec<Rect> {
    full.map(|(w, h)| Rect::new(0, 0, w, h)).into_iter().chain(named.iter().map(|r| r.rect)).collect()
}

pub fn run(a: DatasetArgs) -> Result<()> {
    if a.no_full && a.rois.is_empty() {
        anyhow::bail!("--no-full needs at least one --roi");
    }
    let out =
        a.out.clone().unwrap_or_else(|| PathBuf::from("fixtures/local").join(format!("session-{}", unix_ms() / 1000)));
    std::fs::create_dir_all(&out).with_context(|| format!("creating {}", out.display()))?;

    let mut cap = a.target.open()?;
    let mut full_size = (!a.no_full).then(|| cap.client_size());
    cap.set_rois(&rects(full_size, &a.rois))?;

    let running = Arc::new(AtomicBool::new(true));
    {
        let r = running.clone();
        ctrlc::set_handler(move || r.store(false, Ordering::Relaxed))?;
    }
    let mut hotkey = Hotkey::spawn(&a.hotkey)?;
    let interval = a.interval_ms.map(Duration::from_millis);
    let mut next_auto = interval.map(|i| Instant::now() + i);

    eprintln!(
        "saving to {} — press {} to capture{}, Ctrl+C to stop",
        out.display(),
        a.hotkey,
        interval.map(|i| format!(" (and every {} ms)", i.as_millis())).unwrap_or_default()
    );

    let mut saved = 0u32;
    while running.load(Ordering::Relaxed) && a.count.is_none_or(|c| saved < c) {
        let delivered = cap.next_frame(Duration::from_millis(50))?.map(|f| f.client_size);
        if let (Some(size), Some(full)) = (delivered, full_size)
            && size != full
        {
            eprintln!("client resized to {}x{}", size.0, size.1);
            full_size = Some(size);
            cap.set_rois(&rects(full_size, &a.rois))?;
            continue;
        }

        let mut want = hotkey.take();
        if let (Some(t), Some(i)) = (next_auto, interval)
            && Instant::now() >= t
        {
            want = true;
            next_auto = Some(t + i);
        }
        if !want {
            continue;
        }
        let Some(frame) = cap.last_frame() else {
            eprintln!("no frame captured yet (is the window minimized?)");
            continue;
        };
        saved += 1;
        save(&out, saved, &frame, cap.target(), cap.backend_name(), &a)?;
        eprintln!("saved #{saved:04} (frame {})", frame.frame_id);
    }
    eprintln!("{saved} captures in {}", out.display());
    Ok(())
}

fn save(
    dir: &std::path::Path,
    n: u32,
    frame: &Frame<'_>,
    target: &CaptureTarget,
    backend: &str,
    a: &DatasetArgs,
) -> Result<()> {
    let names = (!a.no_full).then_some("full").into_iter().chain(a.rois.iter().map(|r| r.name.as_str()));
    let mut entries = Vec::new();
    for (name, img) in names.zip(frame.rois) {
        let file = format!("{n:04}_{name}.png");
        write_png(&dir.join(&file), img)?;
        entries.push(serde_json::json!({ "name": name, "rect": img.rect(), "file": file }));
    }
    let target_json = match target {
        CaptureTarget::Window(w) => serde_json::json!({
            "kind": "window", "title": w.title, "process": w.process, "pid": w.pid, "dpi": w.dpi,
        }),
        CaptureTarget::Monitor(m) => serde_json::json!({ "kind": "monitor", "index": m.index, "name": m.name }),
    };
    let sidecar = serde_json::json!({
        "schema": "mapleocr.capture/1",
        "captured_at_unix_ms": unix_ms() as u64,
        "frame_id": frame.frame_id,
        "backend": backend,
        "target": target_json,
        "client_size": frame.client_size,
        "label": a.label,
        "rois": entries,
    });
    std::fs::write(dir.join(format!("{n:04}.json")), serde_json::to_string_pretty(&sidecar)?)?;
    Ok(())
}
