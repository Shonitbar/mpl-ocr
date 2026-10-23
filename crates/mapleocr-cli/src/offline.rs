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
        Ok(Offline { layout, pipeline, templates })
    }
}

struct Offline {
    layout: Layout,
    pipeline: Pipeline,
    templates: Option<AnchorTemplates>,
}

impl Offline {
    fn run(&mut self, img: &BgraImage, dump: Option<(&Path, &str)>) -> Result<(Calibration, Vec<Reading>)> {
        let calib = calibrate(&self.layout, img.view(), self.templates.as_ref());
        let mut out = Vec::new();
        for (i, name) in self.pipeline.roi_names().iter().enumerate() {
            let rect = calib.rois[&**name];
            let crop = img.view().crop(rect);
            let grid = Grid { origin: (rect.x, rect.y), scale: calib.scale };
            if let Some((dir, stem)) = dump {
                save_png(&dir.join(format!("{stem}_{name}_roi.png")), crop)?;
                save_png(&dir.join(format!("{stem}_{name}_ocr.png")), self.pipeline.prepared(i, crop).view())?;
            }
            out.push(self.pipeline.recognize(i, crop, grid, 0, 0)?);
        }
        Ok((calib, out))
    }
}

pub fn describe(v: &Value) -> String {
    match v {
        Value::Map(m) if m.region.is_empty() => m.name.clone(),
        Value::Map(m) => format!("{} / {}", m.region, m.name),
        Value::Text { text } => text.replace('\n', " | "),
        Value::Number { value } => value.to_string(),
        Value::Ratio { current, max } => format!("{current}/{max}"),
        Value::Exp { value, percent } => format!("{value} ({percent:.2}%)"),
        Value::Empty => "(empty)".into(),
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
    #[arg(long)]
    json: bool,
}

pub fn recognize(a: RecognizeArgs) -> Result<()> {
    let mut eng = a.engine.engine()?;
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

#[derive(clap::Subcommand)]
pub enum AtlasCmd {
    /// Teach a `learned:<name>` glyph atlas from one labeled ROI of a saved frame.
    Learn(LearnArgs),
    /// Print an atlas's glyphs.
    Show { name: String },
}

#[derive(Args)]
pub struct LearnArgs {
    /// Full client-area frame (e.g. `*_full.png` from capture-dataset).
    #[arg(long)]
    image: PathBuf,
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

pub fn atlas(cmd: AtlasCmd) -> Result<()> {
    let dir = data_dir()?.join("atlas");
    match cmd {
        AtlasCmd::Learn(a) => {
            let eng = EngineArgs { rois: vec![a.roi.clone()], layout: a.layout }.engine()?;
            let Some(name) = eng.pipeline.learned_font(0).map(str::to_owned) else {
                bail!("ROI {} does not use a learned:<name> glyph font", a.roi);
            };
            let img = load_png(&a.image).with_context(|| format!("loading {}", a.image.display()))?;
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
            let chars: String = atlas.glyphs.iter().map(|g| g.ch).collect();
            println!("{name}: +{added} glyphs, now {} ({chars}) -> {}", atlas.glyphs.len(), path.display());
        }
        AtlasCmd::Show { name } => {
            let atlas = mapleocr_core::Atlas::load(&dir.join(format!("{name}.json")))?;
            for g in &atlas.glyphs {
                println!("'{}' {}x{} drop {}", g.ch, g.bitmap.w, g.bitmap.h, g.drop);
                for row in g.bitmap.bits.chunks(g.bitmap.w.max(1) as usize) {
                    println!("  {}", row.iter().map(|&b| if b { '#' } else { '.' }).collect::<String>());
                }
            }
        }
    }
    Ok(())
}
