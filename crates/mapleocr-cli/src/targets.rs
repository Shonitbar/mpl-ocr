//! `mapleocr targets` and `mapleocr select`.

use anyhow::{Context, Result, bail};
use clap::Args;
use mapleocr_capture::{CaptureTarget, TargetSelector, list_monitors, list_windows};

use crate::TargetArgs;
use crate::config::Config;

#[derive(Args)]
pub struct TargetsArgs {
    /// Print JSON instead of a table.
    #[arg(long)]
    json: bool,
}

#[derive(Args)]
pub struct SelectArgs {
    /// Window index from `mapleocr targets` (saved as process name + title).
    #[arg(long, conflicts_with_all = ["hwnd", "pid", "process", "title", "monitor"])]
    index: Option<usize>,
    #[command(flatten)]
    target: TargetArgs,
}

pub fn run_targets(a: TargetsArgs) -> Result<()> {
    let windows = list_windows();
    let monitors = list_monitors();
    if a.json {
        let v = serde_json::json!({ "windows": windows, "monitors": monitors });
        println!("{}", serde_json::to_string_pretty(&v)?);
        return Ok(());
    }
    let saved = Config::load()?.target;
    let saved_match = saved.as_ref().and_then(|s| s.resolve().ok());

    println!("WINDOWS");
    println!("  {:>3}  {:<10} {:>7}  {:<24} {:>11} {:>4}  title", "#", "hwnd", "pid", "process", "client", "dpi");
    for (i, w) in windows.iter().enumerate() {
        let mark = match &saved_match {
            Some(CaptureTarget::Window(s)) if s.hwnd == w.hwnd => "*",
            _ => " ",
        };
        let size = format!("{}x{}", w.client_size.0, w.client_size.1);
        let min = if w.minimized { " (minimized)" } else { "" };
        println!(
            "{mark} {i:>3}  {:<10} {:>7}  {:<24} {:>11} {:>4}  {}{min}",
            format!("{:#x}", w.hwnd),
            w.pid,
            truncate(&w.process, 24),
            size,
            w.dpi,
            truncate(&w.title, 60)
        );
    }
    println!("MONITORS");
    for m in &monitors {
        let mark = match &saved_match {
            Some(CaptureTarget::Monitor(s)) if s.index == m.index => "*",
            _ => " ",
        };
        let (w, h) = m.size();
        let primary = if m.primary { " primary" } else { "" };
        println!("{mark} m{}  {}  {w}x{h} at ({}, {}){primary}", m.index, m.name, m.bounds.0, m.bounds.1);
    }
    match (&saved, &saved_match) {
        (Some(_), Some(t)) => println!("\n* saved target: {}", t.describe()),
        (Some(s), None) => println!("\nsaved target {s:?} is not running right now"),
        (None, _) => println!("\nno saved target; pick one with `mapleocr select --index N` or `--title ...`"),
    }
    Ok(())
}

fn truncate(s: &str, n: usize) -> String {
    if s.chars().count() <= n { s.to_owned() } else { s.chars().take(n - 1).chain(['…']).collect() }
}

pub fn run_select(a: SelectArgs) -> Result<()> {
    let selector = match a.index {
        Some(i) => {
            let windows = list_windows();
            let w = windows.get(i).with_context(|| format!("no window #{i}; run `mapleocr targets`"))?;
            TargetSelector::for_window(w)
        }
        None => a.target.selector(),
    };
    if selector.is_empty() {
        bail!("give --index N, or at least one of --title/--process/--pid/--hwnd/--monitor");
    }
    let target = selector.resolve().context("selector matches nothing right now")?;
    let mut cfg = Config::load()?;
    cfg.target = Some(selector);
    if let Some(b) = a.target.backend {
        cfg.backend = b.into();
    }
    let p = cfg.save()?;
    println!("selected {} (saved to {})", target.describe(), p.display());
    Ok(())
}
