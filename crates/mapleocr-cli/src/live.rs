//! Live commands against the running game: `calibrate`, `watch`, `read`, and
//! data-pack editing.

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use clap::{Args, Subcommand};
use mapleocr_core::{AnchorTemplates, Layout, MapleOcr, Options, Reading, data_dir};
use mapleocr_imgproc::save_png;
use mapleocr_layout::calibrate;
use mapleocr_semantic::{MapName, MapPack};

use crate::TargetArgs;
use crate::offline::describe;

#[derive(Args)]
pub struct CalibrateArgs {
    #[command(flatten)]
    target: TargetArgs,
    /// Re-cut anchor templates from the current screen. Do this with the UI
    /// in its default arrangement (minimap visible, top-left).
    #[arg(long)]
    bootstrap: bool,
    /// Learn the template of an optional window's anchor (e.g. npc_dialogue)
    /// from a saved full frame where that window is open. No live capture.
    #[arg(long, requires = "image")]
    learn_anchor: Option<String>,
    /// Saved full-client frame for --learn-anchor.
    #[arg(long)]
    image: Option<PathBuf>,
    /// Save each ROI crop here (default fixtures/local/calibrate).
    #[arg(long, default_value = "fixtures/local/calibrate")]
    dump: PathBuf,
}

pub fn calibrate_cmd(a: CalibrateArgs) -> Result<()> {
    if let (Some(name), Some(image)) = (&a.learn_anchor, &a.image) {
        let frame = mapleocr_imgproc::load_png(image).with_context(|| format!("loading {}", image.display()))?;
        let dir = data_dir()?.join("anchors");
        let t = AnchorTemplates::learn_anchor(&dir, &layout()?, name, frame.view())?;
        println!("learned anchor {name} ({} templates in {})", t.images.len(), dir.display());
        return Ok(());
    }
    let mut cap = a.target.open()?;
    let (w, h) = cap.client_size();
    cap.set_rois(&[mapleocr_capture::Rect::new(0, 0, w, h)])?;
    let deadline = Instant::now() + Duration::from_secs(5);
    let frame = loop {
        if let Some(f) = cap.next_frame(Duration::from_millis(100))? {
            break f.rois[0].view().to_owned();
        }
        if Instant::now() > deadline {
            bail!("no frame within 5 s (is the game minimized?)");
        }
    };
    let layout = layout()?;
    let anchors_dir = data_dir()?.join("anchors");
    let templates = match (a.bootstrap, AnchorTemplates::load(&anchors_dir)?) {
        (false, Some(t)) => t,
        _ => {
            let t = AnchorTemplates::bootstrap(&layout, frame.view());
            t.save(&anchors_dir)?;
            println!("anchor templates cut from this screen -> {}", anchors_dir.display());
            t
        }
    };
    let calib = calibrate(&layout, frame.view(), Some(&templates));
    println!("client {}x{}, UI scale x{:.4}", calib.client_size.0, calib.client_size.1, calib.scale);
    for (name, fix) in &calib.anchors {
        let score = fix.score.map(|s| format!("{s:.3}")).unwrap_or_else(|| "-".into());
        println!("anchor {name:<18} {:?} at ({}, {}) score {score}", fix.source, fix.rect.x, fix.rect.y);
    }
    std::fs::create_dir_all(&a.dump)?;
    for (name, r) in &calib.rois {
        let p = a.dump.join(format!("{name}.png"));
        save_png(&p, frame.view().crop(*r))?;
        println!("roi    {name:<18} x {} y {} w {} h {} -> {}", r.x, r.y, r.w, r.h, p.display());
    }
    if !calib.all_matched() {
        println!(
            "\nwarning: some anchors fell back to default positions; if the UI is in its default place, rerun with --bootstrap"
        );
    }
    Ok(())
}

fn layout() -> Result<Layout> {
    let user = data_dir()?.join("layout.yaml");
    Ok(if user.exists() { Layout::load(&user)? } else { Layout::builtin() })
}

fn start(target: &TargetArgs, rois: &[String]) -> Result<MapleOcr> {
    let mut opts = Options::from_config()?;
    opts.selector = target.selector();
    opts.backend = target.backend.map(Into::into);
    opts.rois = (!rois.is_empty()).then(|| rois.to_vec());
    let ocr = MapleOcr::start(opts).context("starting mapleocr (run `mapleocr targets` / `mapleocr select`)")?;
    let s = ocr.status();
    eprintln!("capturing {} via {}", s.target, s.backend);
    Ok(ocr)
}

fn line(r: &Reading) -> String {
    format!(
        "{:<10} {:<45} conf {:.2}{:<7} {:<7} recog {:>5.1} ms  capture->publish {:>6.1} ms  frame {}",
        &*r.roi,
        describe(&r.value),
        r.confidence,
        if r.snapped { " (pack)" } else { "" },
        r.engine,
        r.ms,
        r.latency_ms(),
        r.frame_id
    )
}

#[derive(Args)]
pub struct WatchArgs {
    #[command(flatten)]
    target: TargetArgs,
    /// ROIs to watch (default: all).
    rois: Vec<String>,
    /// Print JSON lines.
    #[arg(long)]
    json: bool,
    /// Stop after this many seconds (default: until Ctrl+C).
    #[arg(long)]
    seconds: Option<f64>,
}

pub fn watch(a: WatchArgs) -> Result<()> {
    let ocr = start(&a.target, &a.rois)?;
    let running = Arc::new(AtomicBool::new(true));
    {
        let r = running.clone();
        ctrlc::set_handler(move || r.store(false, Ordering::Relaxed))?;
    }
    let names: Vec<String> = ocr.roi_names().iter().map(|n| n.to_string()).collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let json = a.json;
    ocr.subscribe(&refs, move |r| {
        if json {
            println!("{}", serde_json::to_string(r).unwrap_or_default());
        } else {
            println!("{}", line(r));
        }
    })?;
    let (cpu0, t0) = (crate::bench::cpu_time(), Instant::now());
    let mut shown_calibration = false;
    let deadline = a.seconds.map(|s| Instant::now() + Duration::from_secs_f64(s));
    while running.load(Ordering::Relaxed) && deadline.is_none_or(|d| Instant::now() < d) {
        std::thread::sleep(Duration::from_millis(200));
        let s = ocr.status();
        if !shown_calibration && let Some(c) = &s.calibration {
            shown_calibration = true;
            eprintln!(
                "calibrated: client {}x{}, scale x{:.4}, anchors {}",
                c.client_size.0,
                c.client_size.1,
                c.scale,
                if s.anchors_matched { "matched" } else { "at default positions" }
            );
        }
        if let Some(e) = &s.last_error {
            eprintln!("error: {e}");
        }
    }
    let s = ocr.status();
    let cpu = (crate::bench::cpu_time() - cpu0).as_secs_f64() / t0.elapsed().as_secs_f64() * 100.0;
    let secs = t0.elapsed().as_secs_f64();
    let pct = |ms: f64| ms / 10.0 / secs;
    eprintln!(
        "time per thread, % of one core: GPU staging {:.1}%, capture dispatch {:.1}%",
        pct(s.stage_ms),
        pct(s.dispatch_ms)
    );
    if s.scouts > 0 {
        eprintln!(
            "  detectors: {:.1}% of one core ({} scouts, {:.1} ms each)",
            pct(s.detect_ms),
            s.scouts,
            s.detect_ms / s.scouts as f64
        );
    }
    let mut by_roi = s.recog_ms.clone();
    by_roi.sort_by(|a, b| b.1.total_cmp(&a.1));
    for (name, ms, n) in by_roi.iter().filter(|r| r.2 > 0) {
        eprintln!("  recognize {name:<16} {:>5.1}%  ({n} jobs, {:.2} ms avg)", pct(*ms), ms / *n as f64);
    }
    eprintln!(
        "{} frames in {:.0} s, {} recognition jobs ({} cache hits), CPU {cpu:.2}% of one core",
        s.frames,
        t0.elapsed().as_secs_f64(),
        s.jobs,
        s.cache_hits
    );
    ocr.stop();
    Ok(())
}

#[derive(Args)]
pub struct ServeArgs {
    #[command(flatten)]
    target: TargetArgs,
    /// ROIs to serve (default: all).
    rois: Vec<String>,
    /// Port on 127.0.0.1.
    #[arg(long, default_value_t = 8765)]
    port: u16,
    /// Also accept browser requests from this origin, e.g. https://myapp.example (repeatable).
    #[arg(long = "allow-origin")]
    allow_origins: Vec<String>,
}

/// Serve readings over HTTP + WebSocket until Ctrl+C.
pub fn serve(a: ServeArgs) -> Result<()> {
    let ocr = std::sync::Arc::new(start(&a.target, &a.rois)?);
    let addr = std::net::SocketAddr::from(([127, 0, 0, 1], a.port));
    mapleocr_server::run_blocking(ocr, addr, a.allow_origins)
}

#[derive(Args)]
pub struct ReadArgs {
    #[command(flatten)]
    target: TargetArgs,
    roi: String,
    #[arg(long, default_value_t = 5.0)]
    timeout_s: f64,
}

pub fn read(a: ReadArgs) -> Result<()> {
    let ocr = start(&a.target, std::slice::from_ref(&a.roi))?;
    let deadline = Instant::now() + Duration::from_secs_f64(a.timeout_s);
    while Instant::now() < deadline {
        if let Some(r) = ocr.read(&a.roi)? {
            println!("{}", serde_json::to_string_pretty(&*r)?);
            ocr.stop();
            return Ok(());
        }
        std::thread::sleep(Duration::from_millis(10));
    }
    let err = ocr.status().last_error.clone();
    ocr.stop();
    bail!("no reading for {} within {} s{}", a.roi, a.timeout_s, err.map(|e| format!(" ({e})")).unwrap_or_default())
}

#[derive(Args)]
pub struct TooltipArgs {
    /// Full client-area frame.
    #[arg(long)]
    image: PathBuf,
    /// Print the binarized native-grid pixels of each tooltip.
    #[arg(long)]
    debug: bool,
    /// With --debug: which ink extraction to print (0-5).
    #[arg(long, default_value_t = 2)]
    variant: usize,
    /// Glyph font spec to read with (default: Arial 12 + 12 bold + 14 bold + 13 bold).
    #[arg(long)]
    font: Option<String>,
}

/// Development tool: find tooltips in a saved frame and show what the reader sees.
pub fn tooltip_cmd(a: TooltipArgs) -> Result<()> {
    let img = mapleocr_imgproc::load_png(&a.image)?;
    let scale = img.height as f32 / 768.0;
    let dir = data_dir()?.join("atlas");
    let spec = a.font.as_deref().unwrap_or("gdi:Arial:12+gdi:Arial:12:bold+gdi:Arial:14:bold+gdi:Arial:13:bold");
    let atlas = mapleocr_core::font(spec, Some(&dir))?.context("font")?;
    for t in mapleocr_core::tooltip::read_all(img.view(), &atlas, scale) {
        println!("== {:?} at {},{}", t.name, t.rect.x, t.rect.y);
        for l in &t.lines {
            println!("   {l}");
        }
    }
    let bullets = mapleocr_core::tooltip::find_bullets(img.view(), scale);
    println!("{} tooltip(s)", bullets.len());
    for b in bullets {
        let loc = mapleocr_core::tooltip::locate(img.view(), b, scale);
        println!("bullet {:?} box {:?}", b, loc.rect);
        if a.debug {
            let c = mapleocr_core::tooltip::corner(b, scale);
            let r = mapleocr_imgproc::Rect::new(c.0, c.1, 470, 470).clamp_to(img.width, img.height);
            let variants = mapleocr_core::tooltip::prepare_variants(img.view(), r, scale);
            for (i, bm) in variants.iter().enumerate() {
                println!("variant {i}: native {}x{}, {} ink px", bm.w, bm.h, bm.bits.iter().filter(|&&b| b).count());
            }
            let bm = &variants[a.variant.min(variants.len() - 1)];
            for row in bm.bits.chunks(bm.w.max(1) as usize).take(140) {
                println!("{}", row.iter().take(190).map(|&b| if b { '#' } else { '.' }).collect::<String>());
            }
        }
    }
    Ok(())
}

#[derive(Subcommand)]
pub enum WindowCmd {
    /// Learn a movable window's title template from a saved frame where it is open.
    Learn {
        /// Template name from the layout (e.g. item_inventory_title).
        template: String,
        #[arg(long)]
        image: PathBuf,
        /// Title strip as x,y,w,h in client pixels of that frame.
        #[arg(long)]
        rect: crate::NamedRoi,
    },
    /// Save the icon of each inventory slot as a labeled entry in your icon database.
    NameIcon {
        /// Detector name (e.g. item_inventory).
        detector: String,
        #[arg(long)]
        image: PathBuf,
        /// Slot index (row-major) and the item name, e.g. --slot 3=Leaf
        #[arg(long = "slot", value_parser = parse_slot)]
        slots: Vec<(u32, String)>,
    },
}

fn parse_slot(s: &str) -> Result<(u32, String), String> {
    let (i, n) = s.split_once('=').ok_or("expected index=name")?;
    Ok((i.parse().map_err(|e: std::num::ParseIntError| e.to_string())?, n.to_owned()))
}

pub fn window(c: WindowCmd) -> Result<()> {
    let dir = data_dir()?;
    match c {
        WindowCmd::Learn { template, image, rect } => {
            let frame = mapleocr_imgproc::load_png(&image)?;
            let t = AnchorTemplates::learn_rect(&dir.join("anchors"), &layout()?, &template, frame.view(), rect.rect)?;
            println!("learned {template} ({} templates)", t.images.len());
        }
        WindowCmd::NameIcon { detector, image, slots } => {
            let frame = mapleocr_imgproc::load_png(&image)?;
            let layout = layout()?;
            let templates = AnchorTemplates::load(&dir.join("anchors"))?.context("no templates learned yet")?;
            let Some(mapleocr_layout::DetectorSpec::Window { template, .. }) = layout.detectors.get(&detector) else {
                bail!("{detector} is not a window detector");
            };
            let calib = calibrate(&layout, frame.view(), Some(&templates));
            let t = templates.scaled(template, calib.scale).context("template not learned")?;
            let title =
                mapleocr_core::detect::find_window_once(frame.view(), &t).context("window not found in that frame")?;
            let mut db = mapleocr_core::detect::IconDb::load(&dir)?;
            for (i, name) in slots {
                let r =
                    mapleocr_core::detect::slot_rect(&layout, &detector, title, calib.scale, i).context("no grid")?;
                let icon = frame.view().crop(mapleocr_imgproc::Rect::new(r.x, r.y, r.w, r.h * 13 / 20));
                let hash = mapleocr_core::detect::icon_hash(icon);
                db.icons.retain(|e| e.hash != hash);
                db.icons.push(mapleocr_core::detect::IconEntry { hash: hash.clone(), name: name.clone() });
                println!("slot {i}: {name} = {hash}");
            }
            db.schema = mapleocr_core::detect::ICON_SCHEMA.into();
            db.save(&dir)?;
        }
    }
    Ok(())
}

#[derive(Subcommand)]
pub enum PackCmd {
    /// Add a map to your local map pack (used to snap OCR results).
    AddMap { region: String, name: String },
    /// List your local map pack.
    ListMaps,
}

pub fn pack(c: PackCmd) -> Result<()> {
    let path = data_dir()?.join("packs").join("maps.json");
    let mut p = MapPack::load_or_default(&path)?;
    match c {
        PackCmd::AddMap { region, name } => {
            let added = p.add(MapName { region, name });
            p.save(&path)?;
            println!(
                "{} ({} maps in {})",
                if added { "added" } else { "already present" },
                p.maps.len(),
                path.display()
            );
        }
        PackCmd::ListMaps => {
            for m in &p.maps {
                println!("{} / {}", m.region, m.name);
            }
            println!("{} maps in {}", p.maps.len(), path.display());
        }
    }
    Ok(())
}
