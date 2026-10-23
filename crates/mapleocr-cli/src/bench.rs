//! `mapleocr bench capture`: live capture latency and CPU cost.
//!
//! Per delivered frame we record three intervals on the QPC clock:
//! - `present→acquired`: compositor/present until our thread holds the frame
//! - `acquired→ready`:   GPU ROI crop + map into CPU buffers (our cost)
//! - `present→ready`:    end-to-end capture latency
//!
//! WGC only delivers frames when the window content changes, so run this
//! while the game is animating.

use std::time::{Duration, Instant};

use anyhow::Result;
use clap::{Args, Subcommand};
use mapleocr_capture::Rect;
use windows::Win32::Foundation::FILETIME;
use windows::Win32::System::Threading::{GetCurrentProcess, GetProcessTimes};

use crate::{NamedRoi, TargetArgs};

#[derive(Subcommand)]
pub enum BenchCmd {
    /// Capture latency and CPU use against the live target.
    Capture(CaptureBenchArgs),
    /// Recognition accuracy and latency on labeled saved frames.
    Fixtures(crate::offline::FixtureBenchArgs),
}

#[derive(Args)]
pub struct CaptureBenchArgs {
    #[command(flatten)]
    target: TargetArgs,
    /// How long to measure.
    #[arg(long, default_value_t = 10)]
    seconds: u64,
    /// Number of synthetic 200x24 ROIs laid out in a grid (ignored with --roi or --full).
    #[arg(long, default_value_t = 8)]
    rois: usize,
    /// Explicit ROI name=x,y,w,h (repeatable).
    #[arg(long = "roi")]
    named: Vec<NamedRoi>,
    /// One ROI covering the whole client area (worst case).
    #[arg(long)]
    full: bool,
    /// Print JSON.
    #[arg(long)]
    json: bool,
}

pub fn run(cmd: BenchCmd) -> Result<()> {
    match cmd {
        BenchCmd::Capture(a) => capture(a),
        BenchCmd::Fixtures(a) => crate::offline::bench_fixtures(a),
    }
}

pub(crate) fn cpu_time() -> Duration {
    let (mut c, mut e, mut k, mut u) = Default::default();
    // SAFETY: valid out-pointers for the current-process pseudo handle.
    unsafe { GetProcessTimes(GetCurrentProcess(), &mut c, &mut e, &mut k, &mut u) }.expect("GetProcessTimes");
    let ft = |f: FILETIME| (f.dwHighDateTime as u64) << 32 | f.dwLowDateTime as u64;
    Duration::from_nanos((ft(k) + ft(u)) * 100)
}

#[derive(Clone, Copy, serde::Serialize)]
struct Stats {
    p50_us: f64,
    p95_us: f64,
    p99_us: f64,
    max_us: f64,
}

fn stats(v: &mut [f64]) -> Option<Stats> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let p = |q: f64| v[((v.len() - 1) as f64 * q).round() as usize];
    Some(Stats { p50_us: p(0.50), p95_us: p(0.95), p99_us: p(0.99), max_us: v[v.len() - 1] })
}

fn capture(a: CaptureBenchArgs) -> Result<()> {
    let mut cap = a.target.open()?;
    let (cw, ch) = cap.client_size();
    let rects: Vec<Rect> = if a.full {
        vec![Rect::new(0, 0, cw, ch)]
    } else if !a.named.is_empty() {
        a.named.iter().map(|r| r.rect).collect()
    } else {
        let cols = (cw / 220).max(1) as usize;
        (0..a.rois).map(|i| Rect::new(10 + (i % cols) as i32 * 220, 10 + (i / cols) as i32 * 34, 200, 24)).collect()
    };
    cap.set_rois(&rects)?;
    let roi_px: u64 = rects.iter().map(|r| r.w as u64 * r.h as u64).sum();

    // Warm up (first frame, driver allocations), then measure.
    let warm_until = Instant::now() + Duration::from_millis(500);
    while Instant::now() < warm_until {
        cap.next_frame(Duration::from_millis(100))?;
    }

    let cap_hint = a.seconds as usize * 240;
    let (mut e2e, mut wake, mut stage) =
        (Vec::with_capacity(cap_hint), Vec::with_capacity(cap_hint), Vec::with_capacity(cap_hint));
    let cpu0 = cpu_time();
    let t0 = Instant::now();
    let end = t0 + Duration::from_secs(a.seconds);
    while Instant::now() < end {
        if let Some(f) = cap.next_frame(Duration::from_millis(100))? {
            e2e.push(f.ready.micros_since(f.present));
            wake.push(f.acquired.micros_since(f.present));
            stage.push(f.ready.micros_since(f.acquired));
        }
    }
    let wall = t0.elapsed();
    let cpu = cpu_time() - cpu0;
    let frames = e2e.len();
    let cpu_pct = cpu.as_secs_f64() / wall.as_secs_f64() * 100.0;
    let fps = frames as f64 / wall.as_secs_f64();
    let intervals = [
        ("present → ready", stats(&mut e2e)),
        ("present → acquired", stats(&mut wake)),
        ("acquired → ready", stats(&mut stage)),
    ];

    let report = serde_json::json!({
        "target": cap.target().describe(),
        "backend": cap.backend_name(),
        "client_size": [cw, ch],
        "rois": rects.len(),
        "roi_pixels": roi_px,
        "seconds": wall.as_secs_f64(),
        "frames": frames,
        "fps": fps,
        "cpu_pct_of_one_core": cpu_pct,
        "present_to_ready": intervals[0].1,
        "present_to_acquired": intervals[1].1,
        "acquired_to_ready": intervals[2].1,
    });
    if a.json {
        println!("{}", serde_json::to_string_pretty(&report)?);
        return Ok(());
    }
    println!(
        "{} via {} | {} ROIs ({} px) | {frames} frames in {:.1}s = {fps:.1} fps | CPU {cpu_pct:.2}% of one core",
        cap.target().describe(),
        cap.backend_name(),
        rects.len(),
        roi_px,
        wall.as_secs_f64()
    );
    println!("{:<22} {:>9} {:>9} {:>9} {:>9}", "interval (µs)", "p50", "p95", "p99", "max");
    for (name, s) in &intervals {
        match s {
            Some(s) => println!("{name:<22} {:>9.1} {:>9.1} {:>9.1} {:>9.1}", s.p50_us, s.p95_us, s.p99_us, s.max_us),
            None => println!("{name:<22} no frames (is the game animating and not minimized?)"),
        }
    }
    Ok(())
}
