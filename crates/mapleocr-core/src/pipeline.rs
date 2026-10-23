//! Per-ROI recognition: preprocess → recognizer tier → semantic parse →
//! [`Reading`]. Shared by the live runtime and offline tools (fixtures).

use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;

use mapleocr_imgproc::{BgraView, Gray, Rect, best_phase, hash_pixels, to_native_phase};
use mapleocr_layout::{Calibration, Layout, RoiSpec};
use mapleocr_recog::preprocess::{self, Crop, Key};
use mapleocr_recog::{Atlas, Reading, Recognition, Recognizer, TextLine, Value};
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

/// Grid origin the phase was found for, and the phase.
type CachedPhase = ((i32, i32), (f64, f64));

/// Load a glyph font spec (see [`Pipeline`] ROI `font`); `None` if it is a
/// `learned:` atlas that does not exist yet.
pub fn font(spec: &str, atlas_dir: Option<&Path>) -> Result<Option<Arc<Atlas>>> {
    load_font(spec, atlas_dir, &mut FontCache::new())
}

/// Resolve a font spec: `gdi:<face>:<px>[:bold|:fakebold]` renders a system font;
/// `learned:<name>` loads `<atlas_dir>/<name>.json` (None if not learned yet).
fn load_font(spec: &str, atlas_dir: Option<&Path>, cache: &mut FontCache) -> Result<Option<Arc<Atlas>>> {
    if let Some(a) = cache.get(spec) {
        return Ok(a.clone());
    }
    if spec.contains('+') {
        let mut parts = Vec::new();
        for part in spec.split('+') {
            match load_font(part, atlas_dir, cache)? {
                Some(a) => parts.push(a),
                None => return Ok(None),
            }
        }
        let refs: Vec<&Atlas> = parts.iter().map(|a| &**a).collect();
        let merged = Some(Arc::new(Atlas::merged(spec, &refs)));
        cache.insert(spec.to_owned(), merged.clone());
        return Ok(merged);
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
            let style = rest.first().copied().unwrap_or("");
            Some(Arc::new(Atlas::from_gdi(face, px, style).map_err(mapleocr_recog::RecogError::from)?))
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

/// Split dialogue lines into the NPC's text and the options below it: the
/// options start after the first vertical gap clearly larger than the line
/// pitch. Wrapped text lines are joined with spaces.
fn split_dialogue(lines: &[TextLine]) -> (String, Vec<String>) {
    let tops: Vec<i32> = lines.iter().map(|l| l.rect.y).collect();
    let mut gaps: Vec<i32> = tops.windows(2).map(|w| w[1] - w[0]).collect();
    gaps.sort_unstable();
    let pitch = gaps.first().copied().unwrap_or(0).max(1);
    let split = tops.windows(2).position(|w| (w[1] - w[0]) as f32 > pitch as f32 * 1.4).map_or(lines.len(), |p| p + 1);
    let text = lines[..split].iter().map(|l| l.text.trim()).collect::<Vec<_>>().join(" ");
    // Options start with a bullet icon, which reads as "?".
    let options = lines[split..]
        .iter()
        .map(|l| l.text.trim().trim_start_matches(['?', ' ']).to_owned())
        .filter(|t| !t.is_empty())
        .collect();
    (text, options)
}

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
    /// Per ROI: native-grid phase found for this grid origin (computing it
    /// costs 16 resamples; it only changes when the geometry does).
    phase: std::cell::RefCell<Vec<Option<CachedPhase>>>,
}

impl Pipeline {
    /// Plans every ROI in `layout`, or only `only` if given.
    /// `atlas_dir` holds learned glyph atlases (`<data_dir>/atlas`).
    pub fn new(layout: &Layout, only: Option<&[String]>, packs: Packs, atlas_dir: Option<&Path>) -> Result<Self> {
        let mut plans = Vec::new();
        let mut fonts = FontCache::new();
        for (name, spec) in &layout.rois {
            let wanted = match only {
                Some(o) => o.iter().any(|n| n == name),
                None => !spec.opt_in,
            };
            if !wanted {
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
        let phase = std::cell::RefCell::new(vec![None; plans.len()]);
        Ok(Self { plans, ocr: None, packs, cache, use_cache: true, phase })
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
        let mut phases = self.phase.borrow_mut();
        let phase = match phases[i] {
            Some((origin, p)) if origin == grid.origin => p,
            _ => {
                let p = best_phase(&ink, grid.origin, grid.scale);
                phases[i] = Some((grid.origin, p));
                p
            }
        };
        to_native_phase(&ink, grid.origin, grid.scale, phase)
    }

    /// For glyph ROIs: the binarized native-grid image the matcher sees, as
    /// `#`/`.` rows. For tuning ROI rects, keys and fonts.
    pub fn glyph_debug(&self, i: usize, img: BgraView<'_>, grid: Grid) -> Option<String> {
        if !matches!(self.plans[i].engine, Engine::Glyph { .. }) {
            return None;
        }
        let c = preprocess::crop_rect(img, self.plans[i].crop);
        let grid = Grid { origin: (grid.origin.0 + c.x, grid.origin.1 + c.y), scale: grid.scale };
        let bm = mapleocr_recog::glyph::Bitmap::binarize(&self.native_ink(i, img.crop(c), grid));
        let rows: Vec<String> = bm
            .bits
            .chunks(bm.w.max(1) as usize)
            .map(|r| r.iter().map(|&b| if b { '#' } else { '.' }).collect())
            .collect();
        Some(rows.join(
            "
",
        ))
    }

    pub fn settle_frames(&self, i: usize) -> u32 {
        self.plans[i].spec.settle_frames
    }

    pub fn min_interval(&self, i: usize) -> std::time::Duration {
        std::time::Duration::from_millis(self.plans[i].spec.min_interval_ms as u64)
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

    /// Recognize ROI `i` from its capture-rect pixels (`calib.capture`),
    /// checking first that an optional window is actually open.
    pub fn recognize_in(
        &mut self,
        i: usize,
        calib: &Calibration,
        capture: BgraView<'_>,
        frame_id: u64,
        captured_at: i64,
    ) -> Result<Reading> {
        let name = self.plans[i].name.clone();
        let (roi, cap) = (calib.rois[&*name], calib.capture[&*name]);
        if let Some(p) = calib.presence.get(&*name)
            && p.is_open(capture) != Some(true)
        {
            return Ok(self.hidden(i, calib, frame_id, captured_at));
        }
        let sub = Rect::new(roi.x - cap.x, roi.y - cap.y, roi.w, roi.h);
        self.recognize_roi(i, calib, capture.crop(sub), frame_id, captured_at)
    }

    /// Recognize ROI `i` from exactly its `calib.rois` pixels (presence, if
    /// any, already checked by the caller).
    pub fn recognize_roi(
        &mut self,
        i: usize,
        calib: &Calibration,
        roi_pixels: BgraView<'_>,
        frame_id: u64,
        captured_at: i64,
    ) -> Result<Reading> {
        let roi = calib.rois[&*self.plans[i].name];
        let grid = Grid { origin: (roi.x, roi.y), scale: calib.scale };
        self.recognize(i, roi_pixels, grid, frame_id, captured_at)
    }

    /// The reading for an optional window that is closed.
    pub fn hidden(&self, i: usize, calib: &Calibration, frame_id: u64, captured_at: i64) -> Reading {
        let name = self.plans[i].name.clone();
        let untrained = calib.presence.get(&*name).is_some_and(|p| p.template.is_none());
        Reading {
            roi: name,
            value: Value::Hidden,
            confidence: 1.0,
            engine: "presence",
            ms: 0.0,
            frame_id,
            captured_at,
            published_at: 0,
            raw: if untrained {
                "anchor template not learned yet (mapleocr calibrate --learn-anchor)".into()
            } else {
                String::new()
            },
            snapped: false,
        }
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
        let (engine, Recognition { lines, boxes, confidence }, key) = match &self.plans[i].engine {
            Engine::Glyph { atlas: Some(atlas), .. } => {
                let atlas = atlas.clone();
                let c = preprocess::crop_rect(img, self.plans[i].crop);
                let (img, grid) =
                    (img.crop(c), Grid { origin: (grid.origin.0 + c.x, grid.origin.1 + c.y), scale: grid.scale });
                let bm = mapleocr_recog::glyph::Bitmap::binarize(&self.native_ink(i, img, grid));
                // Same text pixels -> same reading: skip matching (translucent
                // panels change raw pixels every frame, not the binarized text).
                let packed: Vec<u8> = bm.bits.iter().map(|&b| b as u8).collect();
                let key = hash_pixels(&packed) ^ ((bm.w as u64) << 32 | bm.h as u64);
                if let Some(hit) = self.cache[i].get(&key).filter(|_| self.use_cache) {
                    let ms = t0.elapsed().as_secs_f32() * 1000.0;
                    return Ok(Reading { engine: "cache", ms, frame_id, captured_at, ..hit.clone() });
                }
                let mut r = atlas.read_bitmap(bm);
                // Native-grid boxes -> prepared-image convention used below.
                for b in &mut r.boxes {
                    *b = Rect::new(
                        (b.x as f32 * grid.scale) as i32 + OCR_PAD as i32,
                        (b.y as f32 * grid.scale) as i32 + OCR_PAD as i32,
                        (b.w as f32 * grid.scale) as u32,
                        (b.h as f32 * grid.scale) as u32,
                    );
                }
                ("glyph", r, Some(key))
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
        // Boxes are in prepared-image pixels: undo padding and upscale.
        let text_lines: Vec<TextLine> = lines
            .iter()
            .zip(boxes.iter().copied().chain(std::iter::repeat(Rect::default())))
            .map(|(text, b)| TextLine {
                text: text.clone(),
                rect: Rect::new(
                    grid.origin.0 + ((b.x - OCR_PAD as i32) as f32 / OCR_UPSCALE) as i32,
                    grid.origin.1 + ((b.y - OCR_PAD as i32) as f32 / OCR_UPSCALE) as i32,
                    (b.w as f32 / OCR_UPSCALE) as u32,
                    (b.h as f32 / OCR_UPSCALE) as u32,
                ),
            })
            .collect();
        let (value, confidence, snapped) = match p.spec.parse.as_deref() {
            Some("lines") => (Value::Lines { lines: text_lines }, confidence.unwrap_or(UNCHECKED_CONFIDENCE), false),
            Some("dialogue") => {
                let (text, options) = split_dialogue(&text_lines);
                (Value::Dialogue { text, options }, confidence.unwrap_or(UNCHECKED_CONFIDENCE), false)
            }
            Some("map_name") => match parse_map_name(&lines, self.packs.maps.as_ref()) {
                Some(parsed) => (Value::Map(parsed.value), parsed.confidence, parsed.snapped),
                None => (Value::Empty, 1.0, false),
            },
            _ if raw.trim().is_empty() => (Value::Empty, 1.0, false),
            // An unknown glyph inside a number: report the text, never a guessed number.
            Some("int" | "ratio" | "exp") if raw.contains('?') => (Value::Text { text: raw.clone() }, 0.2, false),
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
            // Plain text: drop unknown glyphs at the ends (borders, icons).
            _ => {
                let text = raw.trim_matches(|c: char| c == '?' || c.is_whitespace()).to_owned();
                (Value::Text { text }, confidence.unwrap_or(UNCHECKED_CONFIDENCE), false)
            }
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
