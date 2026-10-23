//! Offline recognition on saved frames: `mapleocr recognize` and
//! `mapleocr bench fixtures`. Same calibration and pipeline as the live path.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Args;
use mapleocr_core::{AnchorTemplates, Calibration, Grid, Layout, Packs, Pipeline, Reading, Value, data_dir};
use mapleocr_imgproc::{BgraImage, load_png, save_png};
use mapleocr_layout::calibrate;
use serde::Deserialize;

#[derive(Args, Clone)]
pub struct EngineArgs {
    /// Only these ROIs (repeatable). Default: all in the layout.
    #[arg(long = "roi")]
    rois: Vec<String>,
    /// Layout YAML. Default: <data dir>/layout.yaml if present, else built-in.
    #[arg(long)]
    layout: Option<PathBuf>,
}

impl EngineArgs {
    fn layout(&self) -> Result<Layout> {
        let user = data_dir()?.join("layout.yaml");
        Ok(match &self.layout {
            Some(p) => Layout::load(p)?,
            None if user.exists() => Layout::load(&user)?,
            None => Layout::builtin(),
        })
    }

    fn engine(&self) -> Result<Offline> {
        let layout = self.layout()?;
        let dir = data_dir()?;
        let only = (!self.rois.is_empty()).then_some(self.rois.as_slice());
        let pipeline = Pipeline::new(&layout, only, Packs::load(&dir)?, Some(&dir.join("atlas")))?;
        let templates = AnchorTemplates::load(&dir.join("anchors"))?;
        let detectors = mapleocr_core::detect::Detectors::new(&layout, only, &dir)?;
        Ok(Offline { debug: false, layout, pipeline, templates, detectors })
    }
}

struct Offline {
    debug: bool,
    layout: Layout,
    pipeline: Pipeline,
    templates: Option<AnchorTemplates>,
    detectors: mapleocr_core::detect::Detectors,
}

impl Offline {
    fn run(&mut self, img: &BgraImage, dump: Option<(&Path, &str)>) -> Result<(Calibration, Vec<Reading>)> {
        let calib = calibrate(&self.layout, img.view(), self.templates.as_ref());
        let mut out = Vec::new();
        for (i, name) in self.pipeline.roi_names().iter().enumerate() {
            let crop = img.view().crop(calib.rois[&**name]);
            let capture = img.view().crop(calib.capture[&**name]);
            if self.debug {
                let rect = calib.rois[&**name];
                let grid = Grid { origin: (rect.x, rect.y), scale: calib.scale };
                if let Some(pic) = self.pipeline.glyph_debug(i, crop, grid) {
                    eprintln!(
                        "--- {name} (native grid) ---
{pic}"
                    );
                }
            }
            if let Some((dir, stem)) = dump {
                save_png(&dir.join(format!("{stem}_{name}_roi.png")), crop)?;
                save_png(&dir.join(format!("{stem}_{name}_ocr.png")), self.pipeline.prepared(i, crop).view())?;
            }
            out.push(self.pipeline.recognize_in(i, &calib, capture, 0, 0)?);
        }
        // Detectors see the whole frame, like a live scout capture.
        self.detectors.calibrate(&calib, self.templates.as_ref());
        out.extend(self.detectors.run(img.view(), &calib, 0, 0).into_iter().map(|(_, r)| r));
        Ok((calib, out))
    }
}

pub fn describe(v: &Value) -> String {
    match v {
        Value::Map(m) if m.region.is_empty() => m.name.clone(),
        Value::Map(m) => format!("{} / {}", m.region, m.name),
        Value::Text { text } => text.replace('\n', " | "),
        Value::Lines { lines } => lines.iter().map(|l| l.text.as_str()).collect::<Vec<_>>().join(" | "),
        Value::Number { value } => value.to_string(),
        Value::Ratio { current, max } => format!("{current}/{max}"),
        Value::Exp { value, percent } => format!("{value} ({percent:.2}%)"),
        Value::Dialogue { text, options } if options.is_empty() => text.clone(),
        Value::Dialogue { text, options } => format!("{text} || {}", options.join(" | ")),
        Value::Tooltips { items } => {
            items.iter().map(|t| format!("[{}] {}", t.name, t.lines.join(" | "))).collect::<Vec<_>>().join(" || ")
        }
        Value::Window { fields, slots, .. } => {
            let f: Vec<String> = fields.iter().map(|(k, v)| format!("{k}={}", describe(v))).collect();
            let s: Vec<String> = slots
                .iter()
                .map(|s| {
                    format!(
                        "#{}:{}x{}",
                        s.index,
                        s.name.as_deref().unwrap_or(&s.icon[..6]),
                        s.quantity.map_or("?".into(), |q| q.to_string())
                    )
                })
                .collect();
            format!("{} | {} slots: {}", f.join(" "), s.len(), s.join(" "))
        }
        Value::Damage { hits } => hits.iter().map(|h| h.text.clone()).collect::<Vec<_>>().join(", "),
        Value::Empty => "(empty)".into(),
        Value::Hidden => "(hidden)".into(),
    }
}

#[derive(Args)]
pub struct RecognizeArgs {
    /// Full client-area frames (e.g. `*_full.png` from capture-dataset).
    #[arg(required = true)]
    images: Vec<PathBuf>,
    #[command(flatten)]
    engine: EngineArgs,
    /// Save each ROI crop and the image the recognizer saw into this directory.
    #[arg(long)]
    dump: Option<PathBuf>,
    /// Print what the glyph matcher sees (binarized native-grid pixels).
    #[arg(long)]
    debug: bool,
    #[arg(long)]
    json: bool,
}

pub fn recognize(a: RecognizeArgs) -> Result<()> {
    let mut eng = a.engine.engine()?;
    eng.debug = a.debug;
    if let Some(d) = &a.dump {
        std::fs::create_dir_all(d)?;
    }
    for path in &a.images {
        let img = load_png(path).with_context(|| format!("loading {}", path.display()))?;
        let stem = path.file_stem().and_then(|s| s.to_str()).unwrap_or("frame");
        let (calib, readings) = eng.run(&img, a.dump.as_deref().map(|d| (d, stem)))?;
        if a.json {
            println!(
                "{}",
                serde_json::to_string(
                    &serde_json::json!({ "image": path, "calibration": calib, "readings": readings })
                )?
            );
            continue;
        }
        for r in &readings {
            println!(
                "{}  {:<10} {:<45} conf {:.2}{}  {:>6.1} ms",
                path.display(),
                &*r.roi,
                describe(&r.value),
                r.confidence,
                if r.snapped { " (pack)" } else { "" },
                r.ms
            );
        }
    }
    Ok(())
}

/// `fixtures/local/labels.json`:
/// ```json
/// {"schema": "mapleocr.labels/1", "frames": [
///   {"image": "session/0001_full.png", "train": false,
///    "map_name": {"region": "Victoria Road", "name": "Henesys"},
///    "expect": {"hp": "504/554", "exp": "554 (7.86%)", "job": "Rogue"}}]}
/// ```
/// `expect` values compare against the reading's display form. Frames with
/// `train: true` were used to learn atlases; `--held-out` skips them.
#[derive(Deserialize)]
struct Labels {
    schema: String,
    frames: Vec<LabeledFrame>,
}

#[derive(Deserialize)]
struct LabeledFrame {
    image: PathBuf,
    #[serde(default)]
    train: bool,
    map_name: Option<mapleocr_semantic::MapName>,
    #[serde(default)]
    expect: std::collections::BTreeMap<String, String>,
}

#[derive(Args)]
pub struct FixtureBenchArgs {
    /// Labels file (mapleocr.labels/1); image paths are relative to it.
    #[arg(long, default_value = "fixtures/local/labels.json")]
    labels: PathBuf,
    #[command(flatten)]
    engine: EngineArgs,
    /// Ignore data packs (measure raw recognizer accuracy).
    #[arg(long)]
    no_packs: bool,
    /// Skip frames marked `train: true` (used to learn atlases).
    #[arg(long)]
    held_out: bool,
}

#[derive(Default)]
struct RoiStats {
    n: u32,
    ok: u32,
    ms: Vec<f32>,
    engines: std::collections::BTreeSet<&'static str>,
}

pub fn bench_fixtures(a: FixtureBenchArgs) -> Result<()> {
    let labels: Labels = serde_json::from_str(
        &std::fs::read_to_string(&a.labels).with_context(|| format!("reading {}", a.labels.display()))?,
    )?;
    if labels.schema != "mapleocr.labels/1" {
        bail!("{}: expected schema mapleocr.labels/1", a.labels.display());
    }
    let base = a.labels.parent().unwrap_or(Path::new("."));
    let mut eng = a.engine.engine()?;
    if a.no_packs {
        let layout = eng.layout.clone();
        let dir = data_dir()?.join("atlas");
        let only = (!a.engine.rois.is_empty()).then_some(a.engine.rois.as_slice());
        eng.pipeline = Pipeline::new(&layout, only, Packs::default(), Some(&dir))?;
    }
    eng.pipeline.disable_cache();
    let mut stats: std::collections::BTreeMap<String, RoiStats> = Default::default();
    let mut frames = 0;
    for f in labels.frames.iter().filter(|f| !(a.held_out && f.train)) {
        let mut expect = f.expect.clone();
        if let Some(m) = &f.map_name {
            expect.insert("map_name".into(), describe(&Value::Map(m.clone())));
        }
        if expect.is_empty() {
            continue;
        }
        frames += 1;
        let img = load_png(&base.join(&f.image))?;
        let (_, readings) = eng.run(&img, None)?;
        for r in &readings {
            let Some(want) = expect.get(&*r.roi) else { continue };
            let got = describe(&r.value);
            let st = stats.entry(r.roi.to_string()).or_default();
            st.n += 1;
            st.ms.push(r.ms);
            st.engines.insert(r.engine);
            if &got == want {
                st.ok += 1;
            } else {
                println!("MISS {} {}: got {got:?} (raw {:?}), want {want:?}", f.image.display(), r.roi, r.raw);
            }
        }
    }
    if stats.is_empty() {
        bail!("no labeled ROIs in {}", a.labels.display());
    }
    println!(
        "{frames} frames{}{}",
        if a.held_out { ", held-out only" } else { "" },
        if a.no_packs { ", no packs" } else { "" }
    );
    println!("{:<16} {:>6} {:>9} {:>8} {:>8} {:>8}  engine", "roi", "n", "accuracy", "p50 ms", "p95 ms", "max ms");
    for (roi, st) in &mut stats {
        st.ms.sort_by(f32::total_cmp);
        let p = |q: f32| st.ms[((st.ms.len() - 1) as f32 * q).round() as usize];
        let engines: Vec<&str> = st.engines.iter().copied().collect();
        println!(
            "{roi:<16} {:>6} {:>8.1}% {:>8.2} {:>8.2} {:>8.2}  {}",
            st.n,
            100.0 * st.ok as f32 / st.n as f32,
            p(0.5),
            p(0.95),
            st.ms[st.ms.len() - 1],
            engines.join(",")
        );
    }
    Ok(())
}

fn parse_point(s: &str) -> Result<(i32, i32), String> {
    let (x, y) = s.split_once(',').ok_or("expected x,y")?;
    Ok((
        x.trim().parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
        y.trim().parse().map_err(|e: std::num::ParseIntError| e.to_string())?,
    ))
}

#[derive(clap::Subcommand)]
pub enum AtlasCmd {
    /// Teach a `learned:<name>` glyph atlas from one labeled ROI of a saved frame.
    Learn(LearnArgs),
    /// Teach the damage-number digit sprites from a frame where exactly one
    /// damage number is on screen.
    LearnDamage {
        #[arg(long)]
        image: PathBuf,
        /// The digits shown, e.g. 66.
        #[arg(long)]
        text: String,
        /// Only list the candidates found, do not learn.
        #[arg(long)]
        list: bool,
        /// Pick the candidate nearest this point (x,y) when several are found.
        #[arg(long, value_parser = parse_point)]
        at: Option<(i32, i32)>,
    },
    /// Print a font's glyphs: a learned atlas name or any spec (gdi:Arial:11:bold).
    Show {
        name: String,
        /// Only these characters.
        #[arg(long)]
        chars: Option<String>,
    },
}

#[derive(Args)]
pub struct LearnArgs {
    /// Full client-area frame (e.g. `*_full.png` from capture-dataset).
    /// Without it, the current frame is grabbed from the game.
    #[arg(long)]
    image: Option<PathBuf>,
    #[command(flatten)]
    target: crate::TargetArgs,
    /// ROI whose font is `learned:<name>`.
    #[arg(long)]
    roi: String,
    /// Exactly what the ROI shows, e.g. "[544/554]". Spaces are ignored.
    #[arg(long)]
    text: String,
    /// Layout YAML. Default: <data dir>/layout.yaml if present, else built-in.
    #[arg(long)]
    layout: Option<PathBuf>,
}

/// One full client-area frame from the live game.
pub fn grab_frame(target: &crate::TargetArgs) -> Result<BgraImage> {
    let mut cap = target.open()?;
    let (w, h) = cap.client_size();
    cap.set_rois(&[mapleocr_capture::Rect::new(0, 0, w, h)])?;
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if let Some(f) = cap.next_frame(std::time::Duration::from_millis(100))? {
            return Ok(f.rois[0].view().to_owned());
        }
        if std::time::Instant::now() > deadline {
            bail!("no frame within 5 s (is the game minimized?)");
        }
    }
}

pub fn atlas(cmd: AtlasCmd) -> Result<()> {
    let dir = data_dir()?.join("atlas");
    match cmd {
        AtlasCmd::Learn(a) => {
            let eng = EngineArgs { rois: vec![a.roi.clone()], layout: a.layout }.engine()?;
            let Some(name) = eng.pipeline.learned_font(0).map(str::to_owned) else {
                bail!("ROI {} does not use a learned:<name> glyph font", a.roi);
            };
            let img = match &a.image {
                Some(p) => load_png(p).with_context(|| format!("loading {}", p.display()))?,
                None => grab_frame(&a.target)?,
            };
            let calib = calibrate(&eng.layout, img.view(), eng.templates.as_ref());
            let rect = calib.rois[&a.roi];
            let native = eng.pipeline.native_ink(
                0,
                img.view().crop(rect),
                Grid { origin: (rect.x, rect.y), scale: calib.scale },
            );
            let path = dir.join(format!("{name}.json"));
            let mut atlas =
                if path.exists() { mapleocr_core::Atlas::load(&path)? } else { mapleocr_core::Atlas::new(&name, 0) };
            let added = atlas.learn(&native, &a.text).map_err(|e| anyhow::anyhow!("{}: {e}", a.roi))?;
            atlas.save(&path)?;
            let chars: String = atlas.glyphs.iter().filter(|g| g.ch2.is_none()).map(|g| g.ch).collect();
            println!("{name}: +{added} glyphs, now {} ({chars}) -> {}", atlas.glyphs.len(), path.display());
        }
        AtlasCmd::LearnDamage { image, text, list, at } => {
            let img = load_png(&image).with_context(|| format!("loading {}", image.display()))?;
            let layout = Layout::builtin();
            let calib = calibrate(&layout, img.view(), None);
            let found = mapleocr_core::detect::find_damage(img.view(), &[], calib.scale);
            for r in &found {
                println!("candidate at {},{} {}x{}", r.x, r.y, r.w, r.h);
            }
            if list {
                return Ok(());
            }
            let r = match (found.as_slice(), at) {
                ([r], _) => r,
                (_, Some((x, y))) => found
                    .iter()
                    .min_by_key(|r| (r.x + r.w as i32 / 2 - x).abs() + (r.y + r.h as i32 / 2 - y).abs())
                    .context("no candidates")?,
                _ => bail!("found {} candidates; pass --at x,y to pick one (see --list)", found.len()),
            };
            let ink = mapleocr_core::detect::damage_ink(img.view(), *r);
            let path = dir.join("damage_digits.json");
            let mut atlas = if path.exists() {
                mapleocr_core::Atlas::load(&path)?
            } else {
                mapleocr_core::Atlas::new("damage_digits", 0)
            };
            let added = atlas
                .learn_bitmap(mapleocr_core::Bitmap::from_mask(&ink), &text)
                .map_err(|e| anyhow::anyhow!("{e}"))?;
            atlas.save(&path)?;
            println!("damage_digits: +{added} glyphs -> {}", path.display());
        }
        AtlasCmd::Show { name, chars } => {
            let spec = if name.contains(':') { name.clone() } else { format!("learned:{name}") };
            let atlas = mapleocr_core::font(&spec, Some(&dir))?.with_context(|| format!("no atlas {name}"))?;
            for g in atlas.glyphs.iter().filter(|g| g.ch2.is_none() && chars.as_ref().is_none_or(|c| c.contains(g.ch)))
            {
                println!(
                    "'{}{}' {}x{} drop {}",
                    g.ch,
                    g.ch2.map(String::from).unwrap_or_default(),
                    g.bitmap.w,
                    g.bitmap.h,
                    g.drop
                );
                for row in g.bitmap.bits.chunks(g.bitmap.w.max(1) as usize) {
                    println!("  {}", row.iter().map(|&b| if b { '#' } else { '.' }).collect::<String>());
                }
            }
        }
    }
    Ok(())
}
