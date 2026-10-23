//! Per-ROI recognition: preprocess → recognizer tier → semantic parse →
//! [`Reading`]. Shared by the live runtime and offline tools (fixtures).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use mapleocr_imgproc::{BgraView, Gray, hash_pixels, to_native_aligned};
use mapleocr_layout::{Layout, RoiSpec};
use mapleocr_recog::preprocess::{self, Crop, Key};
use mapleocr_recog::{Atlas, Reading, Recognition, Recognizer, Value};
use mapleocr_semantic::{MapPack, UNCHECKED_CONFIDENCE, parse_exp, parse_int, parse_map_name, parse_ratio};

use crate::{Error, Result};

/// User-built data packs, loaded from `<data_dir>/packs/`.
#[derive(Clone, Debug, Default)]
pub struct Packs {
    pub maps: Option<MapPack>,
}

impl Packs {
    pub fn load(data_dir: &Path) -> Result<Self> {
        let maps_path = data_dir.join("packs").join("maps.json");
        let maps = if maps_path.exists() { Some(MapPack::load(&maps_path)?) } else { None };
        Ok(Self { maps })
    }
}

/// Where a ROI sits on the UI grid: its client-pixel top-left and the UI
/// scale. The glyph tier needs it to resample back to native pixels.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Grid {
    pub origin: (i32, i32),
    pub scale: f32,
}

enum Engine {
    Ocr,
    /// `atlas` is `None` until the user learns it; the ROI then uses OCR.
    Glyph {
        font: String,
        atlas: Option<Arc<Atlas>>,
    },
}

struct RoiPlan {
    name: Arc<str>,
    spec: RoiSpec,
    key: Key,
    crop: Crop,
    engine: Engine,
}

type FontCache = HashMap<String, Option<Arc<Atlas>>>;

/// Resolve a font spec: `gdi:<face>:<px>[:bold]` renders a system font;
/// `learned:<name>` loads `<atlas_dir>/<name>.json` (None if not learned yet).
fn load_font(spec: &str, atlas_dir: Option<&Path>, cache: &mut FontCache) -> Result<Option<Arc<Atlas>>> {
    if let Some(a) = cache.get(spec) {
        return Ok(a.clone());
    }
    let parts: Vec<&str> = spec.split(':').collect();
    let atlas = match parts.as_slice() {
        ["learned", name] => atlas_dir
            .map(|d| d.join(format!("{name}.json")))
            .filter(|p| p.exists())
            .map(|p| Atlas::load(&p))
            .transpose()?
            .map(Arc::new),
        #[cfg(windows)]
        ["gdi", face, px, rest @ ..] => {
            let px: i32 = px.parse().map_err(|_| Error::Config(format!("font {spec:?}: bad size")))?;
            let bold = rest.first() == Some(&"bold");
            Some(Arc::new(Atlas::from_gdi(face, px, bold).map_err(mapleocr_recog::RecogError::from)?))
        }
        _ => {
            return Err(Error::Config(format!("unknown font {spec:?}; use gdi:<face>:<px>[:bold] or learned:<name>")));
        }
    };
    cache.insert(spec.to_owned(), atlas.clone());
    Ok(atlas)
}

// Measured on 20 labeled 1920x1200 frames: no upscale + 16 px padding reads
// all of them (2.7 ms); 8 px padding drops short second lines ("Henesys"),
// and 2x upscale is slower with no accuracy gain.
const OCR_UPSCALE: f32 = 1.0;
const OCR_PAD: u32 = 16;

/// Prepared-image cache entries per ROI before that ROI's cache is reset.
const CACHE_PER_ROI: usize = 256;

pub struct Pipeline {
    plans: Vec<RoiPlan>,
    ocr: Option<Box<dyn Recognizer + Send>>,
    packs: Packs,
    /// Per ROI: hash of the prepared (keyed + cropped) image -> reading.
    /// Pixels outside the text (e.g. animated world next to the minimap)
    /// change the raw ROI hash but not this one, so they never reach OCR.
    cache: Vec<HashMap<u64, Reading>>,
    use_cache: bool,
}

impl Pipeline {
    /// Plans every ROI in `layout`, or only `only` if given.
    /// `atlas_dir` holds learned glyph atlases (`<data_dir>/atlas`).
    pub fn new(layout: &Layout, only: Option<&[String]>, packs: Packs, atlas_dir: Option<&Path>) -> Result<Self> {
        let mut plans = Vec::new();
        let mut fonts = FontCache::new();
        for (name, spec) in &layout.rois {
            if only.is_some_and(|o| !o.iter().any(|n| n == name)) {
                continue;
            }
            let key = Key::parse(spec.key.as_deref()).map_err(|e| Error::Config(format!("ROI {name}: {e}")))?;
            let crop = Crop::parse(spec.crop.as_deref()).map_err(|e| Error::Config(format!("ROI {name}: {e}")))?;
            let engine = match spec.recognizer.as_str() {
                "ocr" => Engine::Ocr,
                "glyph" => {
                    let font =
                        spec.font.clone().ok_or_else(|| Error::Config(format!("ROI {name}: glyph needs a font")))?;
                    let atlas = load_font(&font, atlas_dir, &mut fonts)?;
                    Engine::Glyph { font, atlas }
                }
                other => return Err(Error::Config(format!("ROI {name}: recognizer {other:?} is not available yet"))),
            };
            plans.push(RoiPlan { name: name.as_str().into(), spec: spec.clone(), key, crop, engine });
        }
        if let Some(o) = only
            && let Some(missing) = o.iter().find(|n| !layout.rois.contains_key(n.as_str()))
        {
            return Err(Error::UnknownRoi(missing.clone()));
        }
        let cache = plans.iter().map(|_| HashMap::new()).collect();
        Ok(Self { plans, ocr: None, packs, cache, use_cache: true })
    }

    /// Benchmarks: always run the recognizer.
    pub fn disable_cache(&mut self) {
        self.use_cache = false;
    }

    pub fn roi_names(&self) -> Vec<Arc<str>> {
        self.plans.iter().map(|p| p.name.clone()).collect()
    }

    /// For `learned:<name>` glyph ROIs, the atlas name to learn into.
    pub fn learned_font(&self, i: usize) -> Option<&str> {
        match &self.plans[i].engine {
            Engine::Glyph { font, .. } => font.strip_prefix("learned:"),
            Engine::Ocr => None,
        }
    }

    /// Color-keyed ink of ROI `i`, resampled onto the native UI grid.
    pub fn native_ink(&self, i: usize, img: BgraView<'_>, grid: Grid) -> Gray {
        let key = self.plans[i].key;
        let ink = Gray::from_bgra_with(img, |px| Key::ink(px, key) as f32);
        to_native_aligned(&ink, grid.origin, grid.scale)
    }

    pub fn settle_frames(&self, i: usize) -> u32 {
        self.plans[i].spec.settle_frames
    }

    /// The OCR engine is created lazily, on the thread that recognizes.
    fn ocr(&mut self) -> Result<&mut dyn Recognizer> {
        if self.ocr.is_none() {
            #[cfg(windows)]
            {
                self.ocr = Some(Box::new(mapleocr_recog::WinOcr::new()?));
            }
            #[cfg(not(windows))]
            return Err(mapleocr_recog::RecogError::Unavailable("no OCR engine on this platform".into()).into());
        }
        Ok(self.ocr.as_deref_mut().expect("ocr engine"))
    }

    /// The image the recognizer will see (for debugging and fixtures).
    pub fn prepared(&self, i: usize, img: BgraView<'_>) -> mapleocr_imgproc::BgraImage {
        let p = &self.plans[i];
        preprocess::prepare(img, p.key, p.crop, OCR_PAD, OCR_UPSCALE)
    }

    /// Recognize ROI `i` from its pixels.
    pub fn recognize(
        &mut self,
        i: usize,
        img: BgraView<'_>,
        grid: Grid,
        frame_id: u64,
        captured_at: i64,
    ) -> Result<Reading> {
        let t0 = Instant::now();
        let (engine, Recognition { lines, confidence }, key) = match &self.plans[i].engine {
            Engine::Glyph { atlas: Some(atlas), .. } => {
                let atlas = atlas.clone();
                ("glyph", atlas.read(&self.native_ink(i, img, grid)), None)
            }
            Engine::Glyph { atlas: None, .. } | Engine::Ocr => {
                let prepared = self.prepared(i, img);
                let key = hash_pixels(&prepared.data) ^ ((prepared.width as u64) << 32 | prepared.height as u64);
                if let Some(hit) = self.cache[i].get(&key).filter(|_| self.use_cache) {
                    let ms = t0.elapsed().as_secs_f32() * 1000.0;
                    return Ok(Reading { engine: "cache", ms, frame_id, captured_at, ..hit.clone() });
                }
                let ocr = self.ocr()?;
                (ocr.engine(), ocr.recognize(prepared.view())?, Some(key))
            }
        };
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        // Parsers judge plausibility, the engine judges glyph certainty: keep the lower.
        let engine_conf = confidence.unwrap_or(1.0);
        let p = &self.plans[i];
        let raw = lines.join("\n");
        let (value, confidence, snapped) = match p.spec.parse.as_deref() {
            Some("map_name") => match parse_map_name(&lines, self.packs.maps.as_ref()) {
                Some(parsed) => (Value::Map(parsed.value), parsed.confidence, parsed.snapped),
                None => (Value::Empty, 1.0, false),
            },
            _ if raw.trim().is_empty() => (Value::Empty, 1.0, false),
            Some("int") => match parse_int(&raw) {
                Some(value) => (Value::Number { value }, 0.8, false),
                None => (Value::Text { text: raw.clone() }, 0.2, false),
            },
            Some("ratio") => match parse_ratio(&raw) {
                Some(p) => (Value::Ratio { current: p.value.0, max: p.value.1 }, p.confidence, false),
                None => (Value::Text { text: raw.clone() }, 0.2, false),
            },
            Some("exp") => match parse_exp(&raw) {
                Some(p) => (Value::Exp { value: p.value.0, percent: p.value.1 }, p.confidence, false),
                None => (Value::Text { text: raw.clone() }, 0.2, false),
            },
            _ => (Value::Text { text: raw.clone() }, confidence.unwrap_or(UNCHECKED_CONFIDENCE), false),
        };
        let confidence = confidence.min(engine_conf);
        let reading = Reading {
            roi: p.name.clone(),
            value,
            confidence,
            engine,
            ms,
            frame_id,
            captured_at,
            published_at: 0,
            raw,
            snapped,
        };
        if let Some(key) = key {
            if self.cache[i].len() >= CACHE_PER_ROI {
                self.cache[i].clear();
            }
            self.cache[i].insert(key, reading.clone());
        }
        Ok(reading)
    }
}
