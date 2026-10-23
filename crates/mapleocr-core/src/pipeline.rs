//! Per-ROI recognition: preprocess → recognizer tier → semantic parse →
//! [`Reading`]. Shared by the live runtime and offline tools (fixtures).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use mapleocr_imgproc::{BgraView, hash_pixels};
use mapleocr_layout::{Layout, RoiSpec};
use mapleocr_recog::preprocess::{self, Crop, Key};
use mapleocr_recog::{Reading, Recognition, Recognizer, Value};
use mapleocr_semantic::{MapPack, UNCHECKED_CONFIDENCE, parse_map_name};

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

struct RoiPlan {
    name: Arc<str>,
    spec: RoiSpec,
    key: Key,
    crop: Crop,
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
    pub fn new(layout: &Layout, only: Option<&[String]>, packs: Packs) -> Result<Self> {
        let mut plans = Vec::new();
        for (name, spec) in &layout.rois {
            if only.is_some_and(|o| !o.iter().any(|n| n == name)) {
                continue;
            }
            let key = Key::parse(spec.key.as_deref()).map_err(|e| Error::Config(format!("ROI {name}: {e}")))?;
            let crop = Crop::parse(spec.crop.as_deref()).map_err(|e| Error::Config(format!("ROI {name}: {e}")))?;
            if spec.recognizer != "ocr" {
                return Err(Error::Config(format!(
                    "ROI {name}: recognizer {:?} is not available yet",
                    spec.recognizer
                )));
            }
            plans.push(RoiPlan { name: name.as_str().into(), spec: spec.clone(), key, crop });
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
    pub fn recognize(&mut self, i: usize, img: BgraView<'_>, frame_id: u64, captured_at: i64) -> Result<Reading> {
        let t0 = Instant::now();
        let prepared = self.prepared(i, img);
        let key = hash_pixels(&prepared.data) ^ ((prepared.width as u64) << 32 | prepared.height as u64);
        if let Some(hit) = self.cache[i].get(&key).filter(|_| self.use_cache) {
            let ms = t0.elapsed().as_secs_f32() * 1000.0;
            return Ok(Reading { engine: "cache", ms, frame_id, captured_at, ..hit.clone() });
        }
        let ocr = self.ocr()?;
        let engine = ocr.engine();
        let Recognition { lines, confidence } = ocr.recognize(prepared.view())?;
        let ms = t0.elapsed().as_secs_f32() * 1000.0;
        let p = &self.plans[i];
        let raw = lines.join("\n");
        let (value, confidence, snapped) = match p.spec.parse.as_deref() {
            Some("map_name") => match parse_map_name(&lines, self.packs.maps.as_ref()) {
                Some(parsed) => (Value::Map(parsed.value), parsed.confidence, parsed.snapped),
                None => (Value::Empty, 1.0, false),
            },
            _ if raw.trim().is_empty() => (Value::Empty, 1.0, false),
            _ => (Value::Text { text: raw.clone() }, confidence.unwrap_or(UNCHECKED_CONFIDENCE), false),
        };
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
        if self.cache[i].len() >= CACHE_PER_ROI {
            self.cache[i].clear();
        }
        self.cache[i].insert(key, reading.clone());
        Ok(reading)
    }
}
