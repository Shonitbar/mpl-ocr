//! Glyph matcher: the fast tier for fixed-font UI text and sprite digits.
//!
//! Works on the native UI grid (see `mapleocr_imgproc::to_native`), where the
//! client's 1-bit fonts and digit sprites are pixel-exact. A line is
//! binarized, split into glyph segments by empty columns, and each segment is
//! matched against an [`Atlas`] by IoU with ±1 px tolerance. Segments wider
//! than any glyph (touching letters) are split greedily.
//!
//! Atlases are built on the user's machine: from a system font via GDI
//! ([`Atlas::from_gdi`]) or learned from labeled captures ([`Atlas::learn`]).

use std::path::Path;

use mapleocr_imgproc::Gray;
use serde::{Deserialize, Serialize};

use crate::Recognition;

/// Minimum IoU to accept a glyph from a GDI-rendered font (rasterization
/// differs slightly from the game's). Learned sprite atlases are cut from the
/// game itself and match near-exactly, so they demand more: an unlearned "8"
/// must not pass as a "5".
const ACCEPT_GDI: f32 = 0.7;
const ACCEPT_LEARNED: f32 = 0.85;
/// Ink threshold as a fraction from background to ink level. Measured: 0.4
/// merges neighbouring status digits; 0.5 keeps them apart and still keeps
/// 1-px sprites like the "." in "7.86%".
const BINARIZE_AT: f32 = 0.5;

#[derive(Clone, Debug, PartialEq)]
pub struct Bitmap {
    pub w: u32,
    pub h: u32,
    pub bits: Vec<bool>,
}

impl Bitmap {
    /// Ink where the 0/255 image is above 127 (for color masks, where ink can
    /// cover most of the crop and `binarize`'s sparse-ink assumption fails).
    pub fn from_mask(g: &Gray) -> Bitmap {
        Bitmap { w: g.width, h: g.height, bits: g.data.iter().map(|&v| v > 127.0).collect() }
    }

    #[inline]
    fn at(&self, x: i32, y: i32) -> bool {
        x >= 0
            && y >= 0
            && (x as u32) < self.w
            && (y as u32) < self.h
            && self.bits[(y as u32 * self.w + x as u32) as usize]
    }

    /// Adaptive threshold halfway between background (median) and ink (98th percentile).
    pub fn binarize(g: &Gray) -> Bitmap {
        let mut sorted = g.data.clone();
        sorted.sort_by(f32::total_cmp);
        let (bg, ink) = if sorted.is_empty() {
            (0.0, 0.0)
        } else {
            (sorted[sorted.len() / 2], sorted[((sorted.len() - 1) as f32 * 0.98) as usize])
        };
        // Too little contrast: nothing to read.
        let thr = if ink - bg < 40.0 { f32::INFINITY } else { bg + (ink - bg) * BINARIZE_AT };
        Bitmap { w: g.width, h: g.height, bits: g.data.iter().map(|&v| v > thr).collect() }
    }

    fn crop(&self, x0: u32, y0: u32, w: u32, h: u32) -> Bitmap {
        let mut bits = Vec::with_capacity((w * h) as usize);
        for y in y0..y0 + h {
            for x in x0..x0 + w {
                bits.push(self.at(x as i32, y as i32));
            }
        }
        Bitmap { w, h, bits }
    }

    /// Clear rows that are dashed rules (equal-length runs at an equal
    /// pitch, e.g. the underline of a selected dialogue option), so they do
    /// not merge into the text line above.
    pub fn erase_dashed_rules(&mut self) {
        for y in 0..self.h {
            let row = &self.bits[(y * self.w) as usize..((y + 1) * self.w) as usize];
            let mut runs = Vec::new();
            let mut x = 0usize;
            while x < row.len() {
                if row[x] {
                    let s = x;
                    while x < row.len() && row[x] {
                        x += 1;
                    }
                    runs.push((s, x - s));
                } else {
                    x += 1;
                }
            }
            if runs.len() < 6 {
                continue;
            }
            let len = runs[0].1;
            let pitch = runs[1].0 - runs[0].0;
            let regular = runs.iter().all(|r| r.1.abs_diff(len) <= 1)
                && runs.windows(2).all(|w| (w[1].0 - w[0].0).abs_diff(pitch) <= 1);
            // A rule is one pixel tall: little ink directly above or below it.
            // (Monospaced sprite digits also have regular runs, but stacked.)
            let (mut on, mut touching) = (0, 0);
            for &(x0, n) in &runs {
                for x in x0..x0 + n {
                    on += 1;
                    let above = y > 0 && self.bits[((y - 1) * self.w) as usize + x];
                    let below = y + 1 < self.h && self.bits[((y + 1) * self.w) as usize + x];
                    touching += usize::from(above || below);
                }
            }
            if regular && touching * 5 <= on {
                self.bits[(y * self.w) as usize..((y + 1) * self.w) as usize].fill(false);
            }
        }
    }

    /// Tight bounding box of ink within columns `x0..x1` and rows `y0..y1`.
    fn ink_bbox(&self, x0: u32, x1: u32, y0: u32, y1: u32) -> Option<(u32, u32, u32, u32)> {
        let (mut bx0, mut by0, mut bx1, mut by1) = (u32::MAX, u32::MAX, 0, 0);
        for y in y0..y1 {
            for x in x0..x1 {
                if self.bits[(y * self.w + x) as usize] {
                    (bx0, by0, bx1, by1) = (bx0.min(x), by0.min(y), bx1.max(x), by1.max(y));
                }
            }
        }
        (bx0 != u32::MAX).then_some((bx0, by0, bx1 + 1, by1 + 1))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct Glyph {
    pub ch: char,
    /// Second character for pair glyphs: letters GDI draws touching ("ft", "tt").
    pub ch2: Option<char>,
    pub bitmap: Bitmap,
    /// Bottom ink row relative to the baseline row (0 sits on it, >0 descends).
    pub drop: i32,
}

/// A set of glyphs for one font or sprite set.
#[derive(Clone, Debug, PartialEq)]
pub struct Atlas {
    pub name: String,
    /// A gap of at least this many native pixels between glyphs is a space (0: never).
    pub space: u32,
    pub glyphs: Vec<Glyph>,
    /// Minimum IoU to accept a glyph.
    pub accept: f32,
}

#[derive(Serialize, Deserialize)]
struct AtlasFile {
    schema: String,
    name: String,
    space: u32,
    glyphs: Vec<GlyphFile>,
}

#[derive(Serialize, Deserialize)]
struct GlyphFile {
    ch: char,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    ch2: Option<char>,
    drop: i32,
    /// Rows of `#` (ink) and `.`.
    rows: Vec<String>,
}

pub const ATLAS_SCHEMA: &str = "mapleocr.atlas/1";

impl Atlas {
    pub fn new(name: &str, space: u32) -> Self {
        Self { name: name.into(), space, glyphs: Vec::new(), accept: ACCEPT_LEARNED }
    }

    pub fn load(path: &Path) -> std::io::Result<Self> {
        let f: AtlasFile = serde_json::from_str(&std::fs::read_to_string(path)?)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        if f.schema != ATLAS_SCHEMA {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, format!("schema {:?}", f.schema)));
        }
        let glyphs = f
            .glyphs
            .into_iter()
            .map(|g| {
                let h = g.rows.len() as u32;
                let w = g.rows.first().map_or(0, |r| r.chars().count() as u32);
                let bits = g.rows.iter().flat_map(|r| r.chars().map(|c| c == '#')).collect();
                Glyph { ch: g.ch, ch2: g.ch2, drop: g.drop, bitmap: Bitmap { w, h, bits } }
            })
            .collect();
        Ok(Self { name: f.name, space: f.space, glyphs, accept: ACCEPT_LEARNED })
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let f = AtlasFile {
            schema: ATLAS_SCHEMA.into(),
            name: self.name.clone(),
            space: self.space,
            glyphs: self
                .glyphs
                .iter()
                .map(|g| GlyphFile {
                    ch: g.ch,
                    ch2: g.ch2,
                    drop: g.drop,
                    rows: g
                        .bitmap
                        .bits
                        .chunks(g.bitmap.w.max(1) as usize)
                        .map(|r| r.iter().map(|&b| if b { '#' } else { '.' }).collect())
                        .collect(),
                })
                .collect(),
        };
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        std::fs::write(path, serde_json::to_string_pretty(&f).map_err(std::io::Error::other)?)
    }

    /// Union of several atlases (e.g. a font's regular and bold faces).
    pub fn merged(name: &str, parts: &[&Atlas]) -> Atlas {
        let space = parts.iter().map(|a| a.space).filter(|&s| s > 0).min().unwrap_or(0);
        let glyphs = parts.iter().flat_map(|a| a.glyphs.iter().cloned()).collect();
        let accept = parts.iter().map(|a| a.accept).fold(1.0f32, f32::min);
        Atlas { name: name.into(), space, glyphs, accept }
    }

    /// Widest single-character glyph; wider segments are split.
    fn max_w(&self) -> u32 {
        self.glyphs.iter().filter(|g| g.ch2.is_none()).map(|g| g.bitmap.w).max().unwrap_or(0)
    }

    /// Learn glyphs from one labeled line (`text` without its spaces must have
    /// one character per segment). A character already known is added again
    /// only as a distinct variant. Returns the number of glyphs added, or an
    /// error describing the mismatch.
    pub fn learn(&mut self, native: &Gray, text: &str) -> Result<usize, String> {
        self.learn_bitmap(Bitmap::binarize(native), text)
    }

    /// [`Atlas::learn`] from an already binarized image (e.g. a color mask).
    pub fn learn_bitmap(&mut self, bm: Bitmap, text: &str) -> Result<usize, String> {
        let line = lines(&bm).into_iter().max_by_key(|&(a, b)| b - a).ok_or("no ink found")?;
        let segs = segments(&bm, line);
        let chars: Vec<char> = text.chars().filter(|c| !c.is_whitespace()).collect();
        if segs.len() != chars.len() {
            let mut picture = String::new();
            for y in line.0..line.1 {
                picture.push_str("\n  ");
                picture.extend((0..bm.w).map(|x| if bm.at(x as i32, y as i32) { '#' } else { '.' }));
            }
            let widths: Vec<u32> = segs.iter().map(|s| s.x1 - s.x0).collect();
            return Err(format!(
                "found {} glyph segments (widths {widths:?}) but the label has {} characters:{picture}",
                segs.len(),
                chars.len()
            ));
        }
        let baseline = baseline(&segs);
        // Guard against labels shifted by a glyph that split or merged:
        // a segment that already matches a different character exactly.
        for (s, &ch) in segs.iter().zip(&chars) {
            let bitmap = bm.crop(s.x0, s.y0, s.x1 - s.x0, s.y1 - s.y0);
            if let Some(other) = self
                .glyphs
                .iter()
                .filter(|g| g.ch2.is_none() && g.ch != ch)
                .find(|g| iou_shifted(&g.bitmap, &bitmap) >= 0.95)
            {
                return Err(format!(
                    "segment labeled {ch:?} looks exactly like known {:?}: labels are probably shifted (a glyph split or merged); try another frame",
                    other.ch
                ));
            }
        }
        let mut added = 0;
        for (s, ch) in segs.iter().zip(chars) {
            let bitmap = bm.crop(s.x0, s.y0, s.x1 - s.x0, s.y1 - s.y0);
            // Keep a variant per distinct rendering (sub-pixel phase differs
            // between screen positions); skip near-duplicates.
            let known: Vec<&Glyph> = self.glyphs.iter().filter(|g| g.ch == ch && g.ch2.is_none()).collect();
            if known.iter().map(|g| iou_shifted(&g.bitmap, &bitmap)).fold(0.0f32, f32::max) >= 0.95 {
                continue;
            }
            // A variant of a known character has about the same size.
            if !known.is_empty()
                && !known.iter().any(|g| g.bitmap.w.abs_diff(bitmap.w) <= 1 && g.bitmap.h.abs_diff(bitmap.h) <= 1)
            {
                continue;
            }
            self.glyphs.push(Glyph { ch, ch2: None, bitmap, drop: s.y1 as i32 - 1 - baseline });
            added += 1;
        }
        Ok(added)
    }

    /// Read every text line in a native-grid ink image.
    pub fn read(&self, native: &Gray) -> Recognition {
        self.read_bitmap(Bitmap::binarize(native))
    }

    /// Read an already binarized native-grid image (lets callers cache by bitmap).
    pub fn read_bitmap(&self, mut bm: Bitmap) -> Recognition {
        bm.erase_dashed_rules();
        let (mut out, mut boxes) = (Vec::new(), Vec::new());
        let mut worst = 1.0f32;
        for line in lines(&bm) {
            let (text, conf) = self.read_line(&bm, line);
            // Mostly-unknown lines are icons, borders or world pixels, not text.
            // Lines of mostly punctuation are underlines and specks.
            let shown = text.chars().filter(|c| !c.is_whitespace()).count();
            let unknown = text.chars().filter(|&c| c == '?').count();
            let alnum = text.chars().filter(|c| c.is_alphanumeric()).count();
            if shown > 0 && unknown * 2 <= shown && alnum * 2 >= shown {
                worst = worst.min(conf);
                out.push(text);
                let segs = segments(&bm, line);
                let (x0, x1) = (segs.first().map_or(0, |s| s.x0), segs.last().map_or(0, |s| s.x1));
                boxes.push(mapleocr_imgproc::Rect::new(x0 as i32, line.0 as i32, x1 - x0, line.1 - line.0));
            }
        }
        // Boxes are in native-grid pixels of the input.
        Recognition { confidence: Some(if out.is_empty() { 1.0 } else { worst }), lines: out, boxes }
    }

    fn read_line(&self, bm: &Bitmap, line: (u32, u32)) -> (String, f32) {
        let segs = segments(bm, line);
        let base = baseline(&segs);
        let mut text = String::new();
        let mut sims = Vec::new();
        let mut prev_x1: Option<u32> = None;
        let mut i = 0;
        while i < segs.len() {
            let mut s = segs[i];
            i += 1;
            let gap = prev_x1.map(|px| s.x0 - px);
            let (mut best, mut sim) = self.best(&bm.crop(s.x0, s.y0, s.x1 - s.x0, s.y1 - s.y0), s.y1 as i32 - 1 - base);
            // A glyph with an empty column inside (the game's bold "n" has a
            // notch) arrives as two segments: try them merged.
            // Bold "m"/"h" split the same way and their halves can look like
            // "n"+"r": take the merge whenever it fits better than both halves.
            if let Some(next) = segs.get(i)
                && next.x0 - s.x1 <= 1
            {
                let m = Segment { x0: s.x0, x1: next.x1, y0: s.y0.min(next.y0), y1: s.y1.max(next.y1) };
                let (mb, ms) = self.best(&bm.crop(m.x0, m.y0, m.x1 - m.x0, m.y1 - m.y0), m.y1 as i32 - 1 - base);
                let next_sim = self
                    .best(&bm.crop(next.x0, next.y0, next.x1 - next.x0, next.y1 - next.y0), next.y1 as i32 - 1 - base)
                    .1;
                if ms >= self.accept && (sim < self.accept || ms > sim.max(next_sim)) {
                    (s, best, sim) = (m, mb, ms);
                    i += 1;
                }
            }
            if let Some(gap) = gap
                && self.space > 0
            {
                // Digits like "1" have wide side bearings: need a bigger gap.
                let digits =
                    text.ends_with(|c: char| c.is_ascii_digit()) && best.is_some_and(|g| g.ch.is_ascii_digit());
                if gap >= self.space + if digits { 2 } else { 0 } {
                    text.push(' ');
                }
            }
            prev_x1 = Some(s.x1);
            let wide = s.x1 - s.x0 > self.max_w();
            // Blobs much wider than a few letters, or wide and solid (item
            // icons), are not text: skip the expensive split.
            let (sw, sh) = (s.x1 - s.x0, s.y1 - s.y0);
            let ink = (s.y0..s.y1)
                .flat_map(|y| (s.x0..s.x1).map(move |x| (x, y)))
                .filter(|&(x, y)| bm.at(x as i32, y as i32))
                .count();
            let solid = sw > 2 * self.max_w() && ink * 2 > (sw * sh) as usize;
            let too_wide = sw > 4 * self.max_w() || solid;
            if sim >= self.accept || !wide || too_wide {
                push_glyph(&mut text, best.filter(|_| sim >= self.accept));
                sims.push(sim);
            } else {
                for (g, sim) in self.split(bm, &s, base) {
                    push_glyph(&mut text, g);
                    sims.push(sim);
                }
            }
        }
        let conf = sims.iter().copied().fold(1.0f32, f32::min);
        (text, conf)
    }

    /// Best glyph for a tight segment bitmap.
    fn best(&self, seg: &Bitmap, drop: i32) -> (Option<&Glyph>, f32) {
        let mut best = (None, 0.0f32);
        for g in &self.glyphs {
            let gb = &g.bitmap;
            if gb.w.abs_diff(seg.w) > 1 || gb.h.abs_diff(seg.h) > 1 {
                continue;
            }
            let mut sim = iou_shifted(gb, seg);
            if (g.drop - drop).abs() > 1 {
                sim -= 0.25;
            }
            if sim > best.1 {
                best = (Some(g), sim);
            }
        }
        best
    }

    /// Split a segment wider than any glyph (letters the game draws
    /// touching) into the glyph sequence that best explains it: dynamic
    /// programming over column positions, maximizing width-weighted IoU.
    fn split(&self, bm: &Bitmap, s: &Segment, base: i32) -> Vec<(Option<&Glyph>, f32)> {
        let w = (s.x1 - s.x0) as usize;
        // best[x] = (score, previous x, glyph index or usize::MAX for a skipped empty column)
        let mut best: Vec<Option<(f32, usize, usize)>> = vec![None; w + 1];
        best[0] = Some((0.0, 0, usize::MAX));
        let column_empty = |x: usize| !(s.y0..s.y1).any(|y| bm.at((s.x0 as usize + x) as i32, y as i32));
        for x in 0..w {
            let Some((score, _, _)) = best[x] else { continue };
            if column_empty(x) && best[x + 1].is_none_or(|(b, _, _)| score > b) {
                best[x + 1] = Some((score, x, usize::MAX));
            }
            for (gi, g) in self.glyphs.iter().enumerate() {
                let gw = g.bitmap.w as usize;
                if gw == 0 || x + gw > w {
                    continue;
                }
                let bottom = base + g.drop;
                let top = bottom + 1 - g.bitmap.h as i32;
                let mut sim = 0.0f32;
                for dy in -1..=1 {
                    let mut cand =
                        Bitmap { w: gw as u32, h: g.bitmap.h, bits: Vec::with_capacity(gw * g.bitmap.h as usize) };
                    for y in top + dy..=bottom + dy {
                        for dx in 0..gw {
                            cand.bits.push(bm.at((s.x0 as usize + x + dx) as i32, y));
                        }
                    }
                    sim = sim.max(iou(&g.bitmap, &cand, 0, 0) - if dy == 0 { 0.0 } else { 0.02 });
                }
                if sim < self.accept - 0.1 {
                    continue;
                }
                let cand = score + sim * gw as f32;
                if best[x + gw].is_none_or(|(b, _, _)| cand > b) {
                    best[x + gw] = Some((cand, x, gi));
                }
            }
        }
        if best[w].is_none() {
            return vec![(None, 0.0)];
        }
        let mut out = Vec::new();
        let mut x = w;
        while x > 0 {
            let (score, prev, gi) = best[x].expect("reachable");
            if gi != usize::MAX {
                let gain = score - best[prev].map_or(0.0, |b| b.0);
                out.push((Some(&self.glyphs[gi]), gain / (x - prev) as f32));
            }
            x = prev;
        }
        out.reverse();
        out
    }
}

fn push_glyph(text: &mut String, g: Option<&Glyph>) {
    match g {
        Some(g) => {
            text.push(g.ch);
            text.extend(g.ch2);
        }
        None => text.push('?'),
    }
}

/// IoU of ink with `b` shifted by (dx, dy) relative to `a`.
fn iou(a: &Bitmap, b: &Bitmap, dx: i32, dy: i32) -> f32 {
    let (x0, y0) = (dx.min(0), dy.min(0));
    let x1 = (a.w as i32).max(b.w as i32 + dx);
    let y1 = (a.h as i32).max(b.h as i32 + dy);
    let (mut inter, mut union) = (0u32, 0u32);
    for y in y0..y1 {
        for x in x0..x1 {
            let (p, q) = (a.at(x, y), b.at(x - dx, y - dy));
            inter += u32::from(p && q);
            union += u32::from(p || q);
        }
    }
    if union == 0 { 0.0 } else { inter as f32 / union as f32 }
}

fn iou_shifted(a: &Bitmap, b: &Bitmap) -> f32 {
    let mut best = 0.0f32;
    for dy in -1..=1 {
        for dx in -1..=1 {
            // Small penalty for shifting, so exact alignment wins ties.
            let pen = if dx == 0 && dy == 0 { 0.0 } else { 0.02 };
            best = best.max(iou(a, b, dx, dy) - pen);
        }
    }
    best
}

/// Text lines as row ranges, split by empty rows.
fn lines(bm: &Bitmap) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    let mut start = None;
    for y in 0..=bm.h {
        let ink = y < bm.h && (0..bm.w).any(|x| bm.bits[(y * bm.w + x) as usize]);
        match (ink, start) {
            (true, None) => start = Some(y),
            (false, Some(s)) => {
                // Ignore 1-row specks (borders, underlines).
                if y - s >= 3 {
                    out.push((s, y));
                }
                start = None;
            }
            _ => {}
        }
    }
    out
}

#[derive(Clone, Copy, Debug)]
struct Segment {
    x0: u32,
    x1: u32,
    y0: u32,
    y1: u32,
}

/// Glyph segments: runs of inked columns within the line, with tight boxes.
fn segments(bm: &Bitmap, (ly0, ly1): (u32, u32)) -> Vec<Segment> {
    let mut out = Vec::new();
    let mut start = None;
    for x in 0..=bm.w {
        let ink = x < bm.w && (ly0..ly1).any(|y| bm.bits[(y * bm.w + x) as usize]);
        match (ink, start) {
            (true, None) => start = Some(x),
            (false, Some(s)) => {
                if let Some((x0, y0, x1, y1)) = bm.ink_bbox(s, x, ly0, ly1) {
                    out.push(Segment { x0, x1, y0, y1 });
                }
                start = None;
            }
            _ => {}
        }
    }
    out
}

/// Baseline row: the most common bottom ink row among segments.
fn baseline(segs: &[Segment]) -> i32 {
    let mut counts = std::collections::HashMap::new();
    for s in segs {
        *counts.entry(s.y1 as i32 - 1).or_insert(0u32) += 1;
    }
    counts.into_iter().max_by_key(|&(row, n)| (n, row)).map_or(0, |(row, _)| row)
}

#[cfg(windows)]
mod gdi {
    use windows::Win32::Foundation::{COLORREF, SIZE};
    use windows::Win32::Graphics::Gdi::{
        BI_RGB, BITMAPINFO, BITMAPINFOHEADER, CreateCompatibleDC, CreateDIBSection, CreateFontW, DIB_RGB_COLORS,
        DeleteDC, DeleteObject, GdiFlush, GetTextExtentPoint32W, GetTextMetricsW, HGDIOBJ, NONANTIALIASED_QUALITY,
        SelectObject, SetBkColor, SetTextColor, TEXTMETRICW, TextOutW,
    };
    use windows::core::PCWSTR;

    use super::{Atlas, Bitmap, Glyph};

    /// Characters combined into pair glyphs when GDI draws them touching.
    const PAIR_CHARS: &str = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,'-/()";

    /// Printable ASCII, letters first so ambiguous shapes (`l`/`I`/`|`) resolve to letters.
    const CHARS: &str =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;'\"!?-+=/\\()[]<>{}%&*#@$^_~`|";

    impl Atlas {
        /// Render a system font 1-bit with GDI, exactly as a classic Win32
        /// client draws it. `px` is the em height (LOGFONT lfHeight = -px).
        /// `style`: `""` (regular), `"bold"` (the font's bold face), or
        /// `"fakebold"` (regular drawn twice, 1 px apart, as the game does
        /// for quest titles).
        pub fn from_gdi(face: &str, px: i32, style: &str) -> windows::core::Result<Atlas> {
            let bold = style == "bold";
            let fake_bold = style == "fakebold";
            let name = format!("gdi:{face}:{px}{}{style}", if style.is_empty() { "" } else { ":" });
            let face_w: Vec<u16> = face.encode_utf16().chain([0]).collect();
            // SAFETY: GDI objects are created, used, and released within this block.
            unsafe {
                let dc = CreateCompatibleDC(None);
                let font = CreateFontW(
                    -px,
                    0,
                    0,
                    0,
                    if bold { 700 } else { 400 },
                    0,
                    0,
                    0,
                    Default::default(),
                    Default::default(),
                    Default::default(),
                    NONANTIALIASED_QUALITY,
                    0,
                    PCWSTR(face_w.as_ptr()),
                );
                let old_font = SelectObject(dc, HGDIOBJ(font.0));
                let mut tm = TEXTMETRICW::default();
                let _ = GetTextMetricsW(dc, &mut tm);
                let pad = 4;
                let (w, h) = (tm.tmMaxCharWidth + 2 * pad, tm.tmHeight + 2 * pad);
                let bmi = BITMAPINFO {
                    bmiHeader: BITMAPINFOHEADER {
                        biSize: size_of::<BITMAPINFOHEADER>() as u32,
                        biWidth: w,
                        biHeight: -h,
                        biPlanes: 1,
                        biBitCount: 32,
                        biCompression: BI_RGB.0,
                        ..Default::default()
                    },
                    ..Default::default()
                };
                let mut bits = std::ptr::null_mut();
                let dib = CreateDIBSection(Some(dc), &bmi, DIB_RGB_COLORS, &mut bits, None, 0)?;
                let old_bmp = SelectObject(dc, HGDIOBJ(dib.0));
                SetTextColor(dc, COLORREF(0x00FF_FFFF));
                SetBkColor(dc, COLORREF(0));
                let baseline_row = pad + tm.tmAscent - 1;

                let mut space = SIZE::default();
                let sp: Vec<u16> = " ".encode_utf16().collect();
                let _ = GetTextExtentPoint32W(dc, &sp, &mut space);
                let mut atlas = Atlas::new(&name, (space.cx as u32).max(2));
                atlas.accept = super::ACCEPT_GDI;

                let render = |text: &str| -> Option<(Bitmap, i32, bool)> {
                    std::ptr::write_bytes(bits as *mut u8, 0, (w * h * 4) as usize);
                    let s: Vec<u16> = text.encode_utf16().collect();
                    let _ = TextOutW(dc, pad, pad, &s);
                    let _ = GdiFlush();
                    let px = std::slice::from_raw_parts(bits as *const u8, (w * h * 4) as usize);
                    let mut full = Bitmap {
                        w: w as u32,
                        h: h as u32,
                        bits: px.as_chunks::<4>().0.iter().map(|p| p[1] > 127).collect(),
                    };
                    if fake_bold {
                        for row in full.bits.chunks_mut(w as usize) {
                            for x in (1..row.len()).rev() {
                                row[x] |= row[x - 1];
                            }
                        }
                    }
                    let (x0, y0, x1, y1) = full.ink_bbox(0, w as u32, 0, h as u32)?;
                    // One connected run of inked columns = the game will see one segment.
                    let joined = (x0..x1).all(|x| (y0..y1).any(|y| full.at(x as i32, y as i32)));
                    Some((full.crop(x0, y0, x1 - x0, y1 - y0), y1 as i32 - 1 - baseline_row, joined))
                };
                for ch in CHARS.chars() {
                    if let Some((bitmap, drop, _)) = render(&ch.to_string()) {
                        atlas.glyphs.push(Glyph { ch, ch2: None, bitmap, drop });
                    }
                }
                // Letter pairs GDI draws touching ("ft", "tt", "rv"): segmentation
                // cannot split them by empty columns, so match them whole.
                for a in PAIR_CHARS.chars() {
                    for b in PAIR_CHARS.chars() {
                        if let Some((bitmap, drop, true)) = render(&format!("{a}{b}")) {
                            atlas.glyphs.push(Glyph { ch: a, ch2: Some(b), bitmap, drop });
                        }
                    }
                }
                SelectObject(dc, old_bmp);
                let _ = DeleteObject(HGDIOBJ(dib.0));
                SelectObject(dc, old_font);
                let _ = DeleteObject(HGDIOBJ(font.0));
                let _ = DeleteDC(dc);
                Ok(atlas)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gray_from(rows: &[&str]) -> Gray {
        let h = rows.len() as u32;
        let w = rows[0].len() as u32;
        Gray {
            width: w,
            height: h,
            data: rows.iter().flat_map(|r| r.chars().map(|c| if c == '#' { 255.0 } else { 0.0 })).collect(),
        }
    }

    #[test]
    fn learns_and_reads_sprite_digits() {
        let line = gray_from(&[
            "...........................",
            ".###...#...###..####..#..#.",
            "#...#.##..#...#.#.....#..#.",
            "#...#..#.....#..####..####.",
            "#...#..#....#.......#....#.",
            ".###..###.#####.####.....#.",
            "...........................",
        ]);
        let mut a = Atlas::new("test", 0);
        assert_eq!(a.learn(&line, "01254"), Ok(5));
        let r = a.read(&line);
        assert_eq!(r.lines, vec!["01254".to_string()]);
        assert!(r.confidence.unwrap() > 0.99);
        assert!(a.learn(&line, "0125").is_err());
    }

    #[cfg(windows)]
    #[test]
    fn gdi_atlas_reads_its_own_rendering() {
        let a = Atlas::from_gdi("Arial", 12, "").unwrap();
        assert!(a.glyphs.len() > 80);
        // Compose "Rogue 15" from glyph bitmaps with 1 px gaps and a space.
        let word = "Rogue 15";
        let base = 12i32;
        let mut img = Gray::new(80, 18);
        let mut x = 1u32;
        for ch in word.chars() {
            if ch == ' ' {
                x += a.space + 1;
                continue;
            }
            let g = a.glyphs.iter().find(|g| g.ch == ch && g.ch2.is_none()).unwrap();
            let top = base + g.drop + 1 - g.bitmap.h as i32;
            for gy in 0..g.bitmap.h {
                for gx in 0..g.bitmap.w {
                    if g.bitmap.bits[(gy * g.bitmap.w + gx) as usize] {
                        img.data[((top as u32 + gy) * 80 + x + gx) as usize] = 255.0;
                    }
                }
            }
            x += g.bitmap.w + 1;
        }
        assert_eq!(a.read(&img).lines, vec![word.to_string()]);
    }
}

#[cfg(test)]
mod rule_tests {
    use super::*;

    fn bitmap(rows: &[&str]) -> Bitmap {
        Bitmap {
            w: rows[0].len() as u32,
            h: rows.len() as u32,
            bits: rows.iter().flat_map(|r| r.chars().map(|c| c == '#')).collect(),
        }
    }

    #[test]
    fn erases_dashed_underline_but_not_monospaced_digits() {
        let mut underline = bitmap(&["#.#.#.#...........", "..................", "##.##.##.##.##.##."]);
        underline.erase_dashed_rules();
        assert!(underline.bits[36..].iter().all(|&b| !b), "dashed rule cleared");
        assert!(underline.bits[0], "text row kept");
        // Stacked strokes of digit sprites: regular runs, but not 1 px tall.
        let digits = bitmap(&["#..#..#..#..#..#..", "#..#..#..#..#..#..", "#..#..#..#..#..#.."]);
        let mut d = digits.clone();
        d.erase_dashed_rules();
        assert_eq!(d, digits);
    }
}
