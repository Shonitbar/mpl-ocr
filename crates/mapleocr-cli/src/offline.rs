//! Offline recognition on saved frames: `mapleocr recognize` and
//! `mapleocr bench fixtures`. Same calibration and pipeline as the live path.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use clap::Args;
use mapleocr_core::{AnchorTemplates, Calibration, Layout, Packs, Pipeline, Reading, Value, data_dir};
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
        let pipeline = Pipeline::new(&layout, only, Packs::load(&dir)?)?;
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
            let crop = img.view().crop(calib.rois[&**name]);
            if let Some((dir, stem)) = dump {
                save_png(&dir.join(format!("{stem}_{name}_roi.png")), crop)?;
                save_png(&dir.join(format!("{stem}_{name}_ocr.png")), self.pipeline.prepared(i, crop).view())?;
            }
            out.push(self.pipeline.recognize(i, crop, 0, 0)?);
        }
        Ok((calib, out))
    }
}

pub fn describe(v: &Value) -> String {
    match v {
        Value::Map(m) if m.region.is_empty() => m.name.clone(),
        Value::Map(m) => format!("{} / {}", m.region, m.name),
        Value::Text { text } => text.replace('\n', " | "),
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
/// `{"schema": "mapleocr.labels/1", "frames": [{"image": "...png", "map_name": {"region": "...", "name": "..."}}]}`
#[derive(Deserialize)]
struct Labels {
    schema: String,
    frames: Vec<LabeledFrame>,
}

#[derive(Deserialize)]
struct LabeledFrame {
    image: PathBuf,
    map_name: Option<mapleocr_semantic::MapName>,
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
        eng.pipeline = Pipeline::new(&layout, Some(&["map_name".to_string()]), Packs::default())?;
    }
    eng.pipeline.disable_cache();
    let (mut n, mut exact, mut region_ok, mut name_ok) = (0, 0, 0, 0);
    let mut ms = Vec::new();
    for f in &labels.frames {
        let Some(want) = &f.map_name else { continue };
        let img = load_png(&base.join(&f.image))?;
        let (_, readings) = eng.run(&img, None)?;
        let Some(r) = readings.iter().find(|r| &*r.roi == "map_name") else { bail!("layout has no map_name ROI") };
        n += 1;
        ms.push(r.ms);
        let (ok_r, ok_n) = match &r.value {
            Value::Map(m) => (m.region == want.region, m.name == want.name),
            _ => (false, false),
        };
        region_ok += ok_r as u32;
        name_ok += ok_n as u32;
        exact += (ok_r && ok_n) as u32;
        if !(ok_r && ok_n) {
            println!(
                "MISS {}: got {:?} (raw {:?}), want {} / {}",
                f.image.display(),
                describe(&r.value),
                r.raw,
                want.region,
                want.name
            );
        }
    }
    if n == 0 {
        bail!("no labeled map_name frames in {}", a.labels.display());
    }
    ms.sort_by(f32::total_cmp);
    let p = |q: f32| ms[((ms.len() - 1) as f32 * q).round() as usize];
    let pct = |k: u32| 100.0 * k as f32 / n as f32;
    println!(
        "map_name: {n} frames | exact {:.1}% | region {:.1}% | name {:.1}% | recognize ms p50 {:.1} p95 {:.1} p99 {:.1} max {:.1}{}",
        pct(exact),
        pct(region_ok),
        pct(name_ok),
        p(0.5),
        p(0.95),
        p(0.99),
        ms[ms.len() - 1],
        if a.no_packs { " (no packs)" } else { "" }
    );
    Ok(())
}
