//! Detectors: UI that has no fixed position (tooltips, movable windows,
//! damage numbers), found on a full client frame captured at a low rate.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;
use std::time::{Duration, Instant};

use mapleocr_imgproc::{BgraView, Gray, Rect, best_phase, ncc_peaks, ncc_search, to_native_phase};
use mapleocr_layout::{AnchorTemplates, Calibration, DetectorSpec, GridSpec, Layout};
use mapleocr_recog::glyph::Bitmap;
use mapleocr_recog::preprocess::Key;
use mapleocr_recog::{Atlas, Hit, Reading, Slot, Tooltip, Value};
use mapleocr_semantic::parse_int;
use serde::{Deserialize, Serialize};

use crate::pipeline::{Grid, Packs, Pipeline, font};
use crate::{Error, Result};

/// Minimum NCC for a window's title template.
const WINDOW_THRESHOLD: f32 = 0.8;
/// Full-frame window search at most this often; otherwise re-check the last position.
const FULL_SEARCH_EVERY: Duration = Duration::from_millis(1500);
/// A damage number with the same text near the same place within this time is the same hit.
const HIT_MEMORY: Duration = Duration::from_millis(1500);

/// Local icon database: perceptual hash -> item name (`packs/icons.json`).
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct IconDb {
    pub schema: String,
    pub icons: Vec<IconEntry>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IconEntry {
    pub hash: String,
    pub name: String,
}

pub const ICON_SCHEMA: &str = "mapleocr.pack.icons/1";

impl IconDb {
    pub fn path(data_dir: &Path) -> std::path::PathBuf {
        data_dir.join("packs").join("icons.json")
    }

    pub fn load(data_dir: &Path) -> Result<Self> {
        let p = Self::path(data_dir);
        if !p.exists() {
            return Ok(Self { schema: ICON_SCHEMA.into(), icons: Vec::new() });
        }
        serde_json::from_str(&std::fs::read_to_string(&p)?).map_err(|e| Error::Config(format!("{}: {e}", p.display())))
    }

    pub fn save(&self, data_dir: &Path) -> Result<()> {
        let p = Self::path(data_dir);
        std::fs::create_dir_all(p.parent().expect("has parent"))?;
        std::fs::write(&p, serde_json::to_string_pretty(self).map_err(|e| Error::Config(e.to_string()))?)?;
        Ok(())
    }

    /// Closest name within a summed distance of 24 (of 720 possible) of the descriptor.
    pub fn name(&self, hash: &str) -> Option<&str> {
        self.icons
            .iter()
            .filter_map(|e| descriptor_distance(&e.hash, hash).map(|d| (d, &e.name)))
            .filter(|(d, _)| *d <= 24)
            .min_by_key(|(d, _)| *d)
            .map(|(_, n)| n.as_str())
    }
}

/// Icon descriptor: 4x4 cells of mean R,G,B, each quantized to 4 bits, as 48 hex
/// digits. Colors tell smooth gems apart where gradient hashes cannot.
pub fn icon_hash(img: BgraView<'_>) -> String {
    let (w, h) = (img.width.max(1), img.height.max(1));
    let mut sums = [[0f32; 3]; 16];
    let mut counts = [0f32; 16];
    for y in 0..img.height {
        for (x, px) in img.row(y).as_chunks::<4>().0.iter().enumerate() {
            let c = ((y * 4 / h) * 4 + (x as u32 * 4 / w)) as usize;
            sums[c][0] += px[2] as f32;
            sums[c][1] += px[1] as f32;
            sums[c][2] += px[0] as f32;
            counts[c] += 1.0;
        }
    }
    let mut s = String::with_capacity(48);
    for (c, n) in sums.iter().zip(counts) {
        for v in c {
            s.push(char::from_digit(((v / n.max(1.0)) / 16.0).min(15.0) as u32, 16).unwrap_or('0'));
        }
    }
    s
}

fn descriptor_distance(a: &str, b: &str) -> Option<u32> {
    (a.len() == b.len()).then(|| {
        a.chars().zip(b.chars()).map(|(x, y)| x.to_digit(16).unwrap_or(0).abs_diff(y.to_digit(16).unwrap_or(0))).sum()
    })
}

struct WindowState {
    template: String,
    scaled: Option<Gray>,
    last: Option<Rect>,
    last_full: Option<Instant>,
    grid: Option<GridSpec>,
    quantity: Option<Arc<Atlas>>,
    fields: Pipeline,
    field_names: Vec<String>,
    field_rects: Vec<[f32; 4]>,
}

enum Kind {
    /// Tooltip text is laid out around an icon and stat columns, which a line-based
    /// glyph read cannot separate, so detected boxes go to the OCR engine.
    Tooltip {
        ocr: Option<mapleocr_recog::WinOcr>,
    },
    Window(Box<WindowState>),
    Damage {
        atlas: Option<Arc<Atlas>>,
        recent: Vec<(String, (i32, i32), Instant)>,
    },
}

pub struct Detectors {
    names: Vec<Arc<str>>,
    kinds: Vec<Kind>,
    icons: IconDb,
}

/// Mean color of 4x4 blocks: a 1/4-size RGB image (as three planes).
struct Small {
    w: u32,
    h: u32,
    rgb: Vec<[f32; 3]>,
}

impl Small {
    const F: u32 = 4;

    fn new(img: BgraView<'_>) -> Self {
        let (w, h) = (img.width / Self::F, img.height / Self::F);
        let mut rgb = vec![[0.0f32; 3]; (w * h) as usize];
        for y in 0..h * Self::F {
            let row = img.row(y);
            let out = &mut rgb[((y / Self::F) * w) as usize..((y / Self::F + 1) * w) as usize];
            for (x, px) in row.as_chunks::<4>().0.iter().take((w * Self::F) as usize).enumerate() {
                let o = &mut out[x / Self::F as usize];
                o[0] += px[2] as f32;
                o[1] += px[1] as f32;
                o[2] += px[0] as f32;
            }
        }
        let n = (Self::F * Self::F) as f32;
        rgb.iter_mut().for_each(|p| p.iter_mut().for_each(|c| *c /= n));
        Self { w, h, rgb }
    }

    fn gray(&self) -> Gray {
        Gray {
            width: self.w,
            height: self.h,
            data: self.rgb.iter().map(|p| 0.299 * p[0] + 0.587 * p[1] + 0.114 * p[2]).collect(),
        }
    }
}

/// Bounding boxes of 4-connected components of `mask` (w x h), with pixel counts.
fn components(mask: &[bool], w: u32, h: u32) -> Vec<(Rect, u32)> {
    let mut seen = vec![false; mask.len()];
    let mut out = Vec::new();
    let mut stack = Vec::new();
    for start in 0..mask.len() {
        if !mask[start] || seen[start] {
            continue;
        }
        seen[start] = true;
        stack.push(start);
        let (mut x0, mut y0, mut x1, mut y1, mut n) = (u32::MAX, u32::MAX, 0, 0, 0u32);
        while let Some(i) = stack.pop() {
            let (x, y) = (i as u32 % w, i as u32 / w);
            (x0, y0, x1, y1) = (x0.min(x), y0.min(y), x1.max(x), y1.max(y));
            n += 1;
            let mut push = |j: usize| {
                if mask[j] && !seen[j] {
                    seen[j] = true;
                    stack.push(j);
                }
            };
            if x > 0 {
                push(i - 1);
            }
            if x + 1 < w {
                push(i + 1);
            }
            if y > 0 {
                push(i - w as usize);
            }
            if y + 1 < h {
                push(i + w as usize);
            }
        }
        out.push((Rect::new(x0 as i32, y0 as i32, x1 - x0 + 1, y1 - y0 + 1), n));
    }
    out
}

fn overlap(a: Rect, b: Rect) -> u32 {
    let x0 = a.x.max(b.x);
    let y0 = a.y.max(b.y);
    let x1 = a.right().min(b.right());
    let y1 = a.bottom().min(b.bottom());
    ((x1 - x0).max(0) * (y1 - y0).max(0)) as u32
}

/// Tooltip boxes: the translucent navy fill has a consistent color
/// (R,G ~ 70-85, B ~ 113-135) over any background. On the 1/4 frame, match that
/// color, dilate to bridge text and icons, and keep big components. The
/// 1/4-scale box is approximate; text reading tolerates the slack.
pub fn find_tooltips(small: &SmallFrame, exclude: &[Rect]) -> Vec<Rect> {
    let s = &small.0;
    let mask: Vec<bool> = s
        .rgb
        .iter()
        .map(|&[r, g, b]| {
            r > 50.0 && r < 105.0 && g > 50.0 && g < 105.0 && b > 95.0 && b < 150.0 && b - r >= 25.0 && b - g >= 20.0
        })
        .collect();
    let (w, h) = (s.w as usize, s.h as usize);
    let mut d = mask.clone();
    for _ in 0..3 {
        let prev = d.clone();
        for y in 0..h {
            for x in 0..w {
                let i = y * w + x;
                d[i] = prev[i]
                    || (x > 0 && prev[i - 1])
                    || (x + 1 < w && prev[i + 1])
                    || (y > 0 && prev[i - w])
                    || (y + 1 < h && prev[i + w]);
            }
        }
    }
    let f = Small::F as i32;
    // Trimming the blob to dense rows/columns was tried and clipped titles off
    // small tooltips, so the dilated blob is used as is.
    components(&d, s.w, s.h)
        .into_iter()
        .filter(|(r, _)| {
            // Navy cells inside, in the undilated mask: real tooltips have hundreds.
            let mut n = 0;
            for y in r.y..r.bottom() {
                for x in r.x..r.right() {
                    n += u32::from(mask[y as usize * w + x as usize]);
                }
            }
            r.w >= 30 && r.h >= 14 && n >= 500
        })
        .map(|(r, _)| Rect::new(r.x * f, r.y * f, r.w * f as u32, r.h * f as u32))
        .filter(|r| exclude.iter().all(|e| overlap(*r, *e) * 2 < r.w * r.h))
        .collect()
}

/// A full scout frame reduced to 1/4 size, shared by all detectors.
pub struct SmallFrame(Small);

impl SmallFrame {
    pub fn new(img: BgraView<'_>) -> Self {
        Self(Small::new(img))
    }
}

/// Glyph-read a client rect: key -> native grid -> atlas.
fn read_glyphs(img: BgraView<'_>, rect: Rect, scale: f32, key: Key, atlas: &Atlas) -> Vec<String> {
    let crop = img.crop(rect);
    let ink = Gray::from_bgra_with(crop, |px| Key::ink(px, key) as f32);
    // Grid phase from a small sample (the top-left text area), not the whole box.
    let sample = Gray {
        width: ink.width.min(160),
        height: ink.height.min(48),
        data: (0..ink.height.min(48))
            .flat_map(|y| ink.data[(y * ink.width) as usize..(y * ink.width + ink.width.min(160)) as usize].to_vec())
            .collect(),
    };
    let phase = best_phase(&sample, (rect.x, rect.y), scale);
    atlas.read(&to_native_phase(&ink, (rect.x, rect.y), scale, phase)).lines
}

/// Orange damage digits: per-pixel mask over the world area.
fn damage_mask(px: [u8; 4]) -> bool {
    let [b, g, r, _] = px;
    r >= 200 && b <= 90 && r as i32 - b as i32 >= 140 && g >= 40
}

/// Damage-number candidates: groups of digit-sized orange blobs in a row.
pub fn find_damage(img: BgraView<'_>, exclude: &[Rect], scale: f32) -> Vec<Rect> {
    let (w, h) = (img.width, img.height);
    let mut mask = vec![false; (w * h) as usize];
    for y in 0..h {
        for (x, px) in img.row(y).as_chunks::<4>().0.iter().enumerate() {
            mask[(y * w) as usize + x] = damage_mask(*px);
        }
    }
    for e in exclude {
        let c = e.clamp_to(w, h);
        for y in c.y..c.bottom() {
            mask[(y as u32 * w + c.x as u32) as usize..(y as u32 * w + c.right() as u32) as usize].fill(false);
        }
    }
    // Digit-sized blobs (native 8-30 px tall), not big orange scenery.
    let (min_h, max_h) = ((8.0 * scale) as u32, (30.0 * scale) as u32);
    let mut digits: Vec<Rect> = components(&mask, w, h)
        .into_iter()
        .filter(|(r, n)| r.h >= min_h && r.h <= max_h && r.w >= 3 && r.w <= r.h * 3 / 2 && *n * 5 >= r.w * r.h)
        .map(|(r, _)| r)
        .collect();
    digits.sort_by_key(|r| (r.y / 8, r.x));
    // Group horizontally adjacent digits of similar height into numbers.
    let mut numbers: Vec<Rect> = Vec::new();
    for d in digits {
        match numbers.iter_mut().find(|n| {
            (d.x - n.right()).abs() <= (d.h / 2) as i32
                && (d.y - n.y).abs() <= (d.h / 3) as i32
                && d.h.abs_diff(n.h) <= d.h / 3
        }) {
            Some(n) => {
                let (x0, y0) = (n.x.min(d.x), n.y.min(d.y));
                let (x1, y1) = (n.right().max(d.right()), n.bottom().max(d.bottom()));
                *n = Rect::new(x0, y0, (x1 - x0) as u32, (y1 - y0) as u32);
            }
            None => numbers.push(d),
        }
    }
    numbers
}

/// The damage digit mask of a number's rect, as a 0/255 image for the atlas.
pub fn damage_ink(img: BgraView<'_>, rect: Rect) -> Gray {
    let crop = img.crop(rect.clamp_to(img.width, img.height));
    Gray::from_bgra_with(crop, |px| if damage_mask(px) { 255.0 } else { 0.0 })
}

impl Detectors {
    pub fn new(layout: &Layout, only: Option<&[String]>, data_dir: &Path) -> Result<Self> {
        let atlas_dir = data_dir.join("atlas");
        let (mut names, mut kinds) = (Vec::new(), Vec::new());
        for (name, spec) in &layout.detectors {
            if only.is_some_and(|o| !o.iter().any(|n| n == name)) {
                continue;
            }
            let kind = match spec {
                DetectorSpec::Tooltip { .. } => Kind::Tooltip { ocr: None },
                DetectorSpec::Damage { font: f } => {
                    Kind::Damage { atlas: font(f, Some(&atlas_dir))?, recent: Vec::new() }
                }
                DetectorSpec::Window { template, grid, fields } => {
                    let sub = Layout {
                        schema: layout.schema.clone(),
                        canvas_height: layout.canvas_height,
                        anchors: BTreeMap::new(),
                        rois: fields.iter().map(|(k, f)| (k.clone(), f.as_roi())).collect(),
                        detectors: BTreeMap::new(),
                    };
                    let pipeline = Pipeline::new(&sub, None, Packs::default(), Some(&atlas_dir))?;
                    let field_names = pipeline.roi_names().iter().map(|n| n.to_string()).collect::<Vec<_>>();
                    let field_rects = field_names.iter().map(|n| fields[n].rect).collect();
                    let quantity = match grid {
                        Some(g) => font(&g.quantity_font, Some(&atlas_dir))?,
                        None => None,
                    };
                    Kind::Window(Box::new(WindowState {
                        template: template.clone(),
                        scaled: None,
                        last: None,
                        last_full: None,
                        grid: grid.clone(),
                        quantity,
                        fields: pipeline,
                        field_names,
                        field_rects,
                    }))
                }
            };
            names.push(Arc::<str>::from(name.as_str()));
            kinds.push(kind);
        }
        if let Some(o) = only {
            for n in o {
                if !layout.detectors.contains_key(n) && !layout.rois.contains_key(n) {
                    return Err(Error::UnknownRoi(n.clone()));
                }
            }
        }
        Ok(Self { names, kinds, icons: IconDb::load(data_dir)? })
    }

    pub fn names(&self) -> &[Arc<str>] {
        &self.names
    }

    pub fn is_empty(&self) -> bool {
        self.names.is_empty()
    }

    /// Rescale learned window templates for a new client size.
    pub fn calibrate(&mut self, calib: &Calibration, templates: Option<&AnchorTemplates>) {
        for k in &mut self.kinds {
            if let Kind::Window(w) = k {
                w.scaled = templates.and_then(|t| t.scaled(&w.template, calib.scale));
                w.last = None;
                w.last_full = None;
            }
        }
    }

    /// Run every detector on one full client frame.
    pub fn run(
        &mut self,
        img: BgraView<'_>,
        calib: &Calibration,
        frame_id: u64,
        captured_at: i64,
    ) -> Vec<(usize, Reading)> {
        let small = SmallFrame::new(img);
        let small_gray = small.0.gray();
        let scale = calib.scale;
        // Fixed HUD regions never hold tooltips or damage numbers.
        // (Not optional windows like NPC dialogue: usually closed, and centered.)
        let fixed: Vec<Rect> = calib
            .rois
            .iter()
            .filter(|(n, r)| {
                !calib.presence.contains_key(*n) && r.w * r.h < calib.client_size.0 * calib.client_size.1 / 4
            })
            .map(|(_, r)| *r)
            .collect();
        let mut windows_found: Vec<Rect> = Vec::new();
        let mut out = Vec::new();
        let reading = |i: usize, value: Value, confidence: f32, ms: f32, names: &[Arc<str>]| Reading {
            roi: names[i].clone(),
            value,
            confidence,
            engine: "detector",
            ms,
            frame_id,
            captured_at,
            published_at: 0,
            raw: String::new(),
            snapped: false,
        };
        // Windows first: their rects are excluded from the other detectors.
        for i in 0..self.kinds.len() {
            let t0 = Instant::now();
            let Kind::Window(w) = &mut self.kinds[i] else { continue };
            let found = locate_window(w, img, &small_gray);
            let value = match found {
                Some(rect) => {
                    // Rough window extent below its title, for excluding it from other detectors.
                    windows_found.push(Rect::new(
                        rect.x - (10.0 * scale) as i32,
                        rect.y,
                        (240.0 * scale) as u32,
                        (300.0 * scale) as u32,
                    ));
                    read_window(w, &self.icons, img, rect, scale, frame_id, captured_at)
                }
                None => Value::Hidden,
            };
            out.push((i, reading(i, value, 0.9, t0.elapsed().as_secs_f32() * 1000.0, &self.names)));
        }
        let exclude: Vec<Rect> = fixed.iter().chain(&windows_found).copied().collect();
        let tooltips = if self.kinds.iter().any(|k| matches!(k, Kind::Tooltip { .. })) {
            // Only movable windows are excluded: a tooltip may sit on top of fixed
            // HUD panels (e.g. the quest helper); the 500-navy-cell floor already
            // rejects those panels themselves.
            find_tooltips(&small, &windows_found)
        } else {
            Vec::new()
        };
        for i in 0..self.kinds.len() {
            let t0 = Instant::now();
            match &mut self.kinds[i] {
                Kind::Tooltip { ocr } => {
                    if tooltips.is_empty() {
                        out.push((i, reading(i, Value::Hidden, 0.8, 0.0, &self.names)));
                        continue;
                    }
                    if ocr.is_none() {
                        *ocr = mapleocr_recog::WinOcr::new().ok();
                    }
                    let Some(ocr) = ocr else { continue };
                    let items: Vec<Tooltip> = tooltips
                        .iter()
                        .filter_map(|&r| {
                            let crop = img.crop(r);
                            let prepared = mapleocr_recog::preprocess::prepare(
                                crop,
                                Key::Luma,
                                mapleocr_recog::preprocess::Crop::None,
                                16,
                                1.0,
                            );
                            use mapleocr_recog::Recognizer;
                            let rec = ocr.recognize(prepared.view()).ok()?;
                            // The bullet before the title reads as junk; the name is the
                            // first line with a real word after stripping it.
                            let mut lines = rec
                                .lines
                                .into_iter()
                                .map(|l| l.trim_start_matches(|c: char| !c.is_alphanumeric()).trim().to_owned())
                                .filter(|l| !l.is_empty())
                                .peekable();
                            while lines.peek().is_some_and(|l| l.chars().filter(|c| c.is_alphabetic()).count() < 3) {
                                lines.next();
                            }
                            let name = lines.next().unwrap_or_default();
                            Some(Tooltip { name, lines: lines.collect(), rect: r })
                        })
                        .collect();
                    let v = if items.is_empty() { Value::Hidden } else { Value::Tooltips { items } };
                    out.push((i, reading(i, v, 0.8, t0.elapsed().as_secs_f32() * 1000.0, &self.names)));
                }
                Kind::Damage { atlas, recent } => {
                    let Some(atlas) = atlas else { continue };
                    let ex: Vec<Rect> = exclude.iter().chain(&tooltips).copied().collect();
                    let now = Instant::now();
                    recent.retain(|(_, _, t)| now.duration_since(*t) < HIT_MEMORY);
                    let mut hits = Vec::new();
                    for r in find_damage(img, &ex, scale) {
                        let rec = atlas.read_bitmap(Bitmap::from_mask(&damage_ink(img, r)));
                        let Some(text) = rec.lines.into_iter().next() else { continue };
                        let known = text.chars().filter(|c| c.is_ascii_digit()).count();
                        if known * 2 < text.chars().count() {
                            continue; // mostly unknown shapes: scenery, not a number
                        }
                        let c = (r.x + r.w as i32 / 2, r.y + r.h as i32 / 2);
                        let seen = recent
                            .iter()
                            .any(|(t, p, _)| *t == text && (p.0 - c.0).abs() < 80 && (p.1 - c.1).abs() < 120);
                        recent.push((text.clone(), c, now));
                        if !seen {
                            let value = if text.contains('?') { None } else { parse_int(&text) };
                            hits.push(Hit { value, text, rect: r });
                        }
                    }
                    if !hits.is_empty() {
                        out.push((
                            i,
                            reading(i, Value::Damage { hits }, 0.8, t0.elapsed().as_secs_f32() * 1000.0, &self.names),
                        ));
                    }
                }
                Kind::Window(_) => {}
            }
        }
        out
    }
}

/// Find a window's title: re-check the last position at full resolution, and
/// search the whole 1/4 frame at most every [`FULL_SEARCH_EVERY`].
fn locate_window(w: &mut WindowState, img: BgraView<'_>, small: &Gray) -> Option<Rect> {
    let t = w.scaled.as_ref()?;
    let check = |x: i32, y: i32, slack: i32| -> Option<(Rect, f32)> {
        let area = Rect::new(x - slack, y - slack, t.width + 2 * slack as u32, t.height + 2 * slack as u32)
            .clamp_to(img.width, img.height);
        let g = Gray::from_bgra(img.crop(area));
        let m = ncc_search(&g, t, Rect::new(0, 0, g.width, g.height))?;
        Some((Rect::new(area.x + m.x, area.y + m.y, t.width, t.height), m.score))
    };
    if let Some(last) = w.last
        && let Some((r, s)) = check(last.x, last.y, 2)
        && s >= WINDOW_THRESHOLD
    {
        w.last = Some(r);
        return Some(r);
    }
    if w.last_full.is_some_and(|t| t.elapsed() < FULL_SEARCH_EVERY) {
        w.last = None;
        return None;
    }
    w.last_full = Some(Instant::now());
    // Coarse candidates on the 1/4 frame (ambiguous alone), verified at full size.
    let ts = t.downscale(Small::F);
    let f = Small::F as i32;
    w.last = ncc_peaks(small, &ts, 6)
        .into_iter()
        .filter_map(|m| check(m.x * f, m.y * f, 2 * f))
        .filter(|(_, s)| *s >= WINDOW_THRESHOLD)
        .max_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(r, _)| r);
    w.last
}

fn read_window(
    w: &mut WindowState,
    icons: &IconDb,
    img: BgraView<'_>,
    title: Rect,
    scale: f32,
    frame_id: u64,
    captured_at: i64,
) -> Value {
    let at = |nx: f32, ny: f32, nw: f32, nh: f32| {
        Rect::new(
            title.x + (nx * scale).round() as i32,
            title.y + (ny * scale).round() as i32,
            (nw * scale).round() as u32,
            (nh * scale).round() as u32,
        )
        .clamp_to(img.width, img.height)
    };
    let mut fields = BTreeMap::new();
    for (k, (name, r)) in w.field_names.iter().zip(&w.field_rects).enumerate() {
        let rect = at(r[0], r[1], r[2], r[3]);
        let grid = Grid { origin: (rect.x, rect.y), scale };
        if let Ok(reading) = w.fields.recognize(k, img.crop(rect), grid, frame_id, captured_at) {
            fields.insert(name.clone(), reading.value);
        }
    }
    let mut slots = Vec::new();
    if let Some(g) = &w.grid {
        for row in 0..g.rows {
            for col in 0..g.cols {
                let (x, y) = (g.origin[0] + col as f32 * g.pitch[0], g.origin[1] + row as f32 * g.pitch[1]);
                let slot = at(x, y, g.size[0], g.size[1]);
                let icon = img.crop(Rect::new(slot.x, slot.y, slot.w, slot.h * 13 / 20));
                // Empty slots are flat beige.
                let gray = Gray::from_bgra(icon);
                let n = gray.data.len().max(1) as f32;
                let mean = gray.data.iter().sum::<f32>() / n;
                let sd = (gray.data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt();
                if sd < 10.0 {
                    continue;
                }
                let hash = icon_hash(icon);
                let quantity = w.quantity.as_ref().and_then(|a| {
                    let qr = Rect::new(slot.x, slot.y + (slot.h * 11 / 20) as i32, slot.w * 3 / 4, slot.h * 9 / 20);
                    let lines = read_glyphs(img, qr, scale, Key::White, a);
                    lines.first().filter(|t| !t.contains('?')).and_then(|t| parse_int(t))
                });
                slots.push(Slot {
                    index: row * g.cols + col,
                    name: icons.name(&hash).map(str::to_owned),
                    icon: hash,
                    quantity,
                });
            }
        }
    }
    Value::Window { rect: title, fields, slots }
}

/// Slot rect (client pixels) of a window grid, for teaching tools.
pub fn slot_rect(layout: &Layout, detector: &str, title: Rect, scale: f32, index: u32) -> Option<Rect> {
    let Some(DetectorSpec::Window { grid: Some(g), .. }) = layout.detectors.get(detector) else { return None };
    let (row, col) = (index / g.cols, index % g.cols);
    let (x, y) = (g.origin[0] + col as f32 * g.pitch[0], g.origin[1] + row as f32 * g.pitch[1]);
    Some(Rect::new(
        title.x + (x * scale).round() as i32,
        title.y + (y * scale).round() as i32,
        (g.size[0] * scale).round() as u32,
        (g.size[1] * scale).round() as u32,
    ))
}

/// Locate a window's title once (for teaching tools), searching the whole frame.
pub fn find_window_once(img: BgraView<'_>, template: &Gray) -> Option<Rect> {
    let mut w = WindowState {
        template: String::new(),
        scaled: Some(template.clone()),
        last: None,
        last_full: None,
        grid: None,
        quantity: None,
        fields: Pipeline::new(
            &Layout {
                schema: "mapleocr.layout/1".into(),
                canvas_height: 768.0,
                anchors: BTreeMap::new(),
                rois: BTreeMap::new(),
                detectors: BTreeMap::new(),
            },
            None,
            Packs::default(),
            None,
        )
        .ok()?,
        field_names: Vec::new(),
        field_rects: Vec::new(),
    };
    let small = SmallFrame::new(img).0.gray();
    locate_window(&mut w, img, &small)
}

#[cfg(test)]
mod tests {
    use super::*;
    use mapleocr_imgproc::BgraImage;

    fn paint(img: &mut BgraImage, r: Rect, [rr, g, b]: [u8; 3]) {
        for y in r.y..r.bottom() {
            for x in r.x..r.right() {
                let o = ((y as u32 * img.width + x as u32) * 4) as usize;
                img.data[o..o + 4].copy_from_slice(&[b, g, rr, 255]);
            }
        }
    }

    #[test]
    fn finds_navy_tooltip_but_not_sky_or_small_panels() {
        let mut img = BgraImage::new(800, 600);
        paint(&mut img, Rect::new(0, 0, 800, 600), [20, 79, 220]); // bright blue sky
        paint(&mut img, Rect::new(100, 120, 300, 260), [74, 77, 121]); // tooltip fill
        paint(&mut img, Rect::new(600, 40, 40, 30), [74, 77, 121]); // too small
        let found = find_tooltips(&SmallFrame::new(img.view()), &[]);
        assert_eq!(found.len(), 1, "{found:?}");
        let r = found[0];
        assert!(r.x <= 100 && r.right() >= 400 && r.y <= 120 && r.bottom() >= 380, "{r:?}");
    }

    #[test]
    fn components_and_overlap() {
        let mask = [true, true, false, false, false, false, false, false, false, false, true, true];
        let c = components(&mask, 4, 3);
        assert_eq!(c.len(), 2);
        assert_eq!(overlap(Rect::new(0, 0, 10, 10), Rect::new(5, 5, 10, 10)), 25);
    }

    #[test]
    fn icon_descriptor_tells_colors_apart_and_names_them() {
        let solid = |c: [u8; 3]| {
            let mut i = BgraImage::new(32, 32);
            paint(&mut i, Rect::new(0, 0, 32, 32), c);
            i
        };
        let (red, blue) = (icon_hash(solid([200, 30, 30]).view()), icon_hash(solid([30, 30, 200]).view()));
        assert_ne!(red, blue);
        let db =
            IconDb { schema: ICON_SCHEMA.into(), icons: vec![IconEntry { hash: red.clone(), name: "Ribbon".into() }] };
        assert_eq!(db.name(&red), Some("Ribbon"));
        assert_eq!(db.name(&blue), None);
    }
}
