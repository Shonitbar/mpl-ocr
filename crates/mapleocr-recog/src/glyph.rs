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

/// Minimum IoU to accept a glyph.
const ACCEPT: f32 = 0.7;
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
    drop: i32,
    /// Rows of `#` (ink) and `.`.
    rows: Vec<String>,
}

pub const ATLAS_SCHEMA: &str = "mapleocr.atlas/1";

impl Atlas {
    pub fn new(name: &str, space: u32) -> Self {
        Self { name: name.into(), space, glyphs: Vec::new() }
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
                Glyph { ch: g.ch, drop: g.drop, bitmap: Bitmap { w, h, bits } }
            })
            .collect();
        Ok(Self { name: f.name, space: f.space, glyphs })
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

    fn max_w(&self) -> u32 {
        self.glyphs.iter().map(|g| g.bitmap.w).max().unwrap_or(0)
    }

    /// Learn glyphs from one labeled line (`text` without its spaces must have
    /// one character per segment). Existing characters are kept. Returns the
    /// number of new glyphs, or an error describing the mismatch.
    pub fn learn(&mut self, native: &Gray, text: &str) -> Result<usize, String> {
        let bm = Bitmap::binarize(native);
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
        let mut added = 0;
        for (s, ch) in segs.iter().zip(chars) {
            if self.glyphs.iter().any(|g| g.ch == ch) {
                continue;
            }
            let bitmap = bm.crop(s.x0, s.y0, s.x1 - s.x0, s.y1 - s.y0);
            self.glyphs.push(Glyph { ch, bitmap, drop: s.y1 as i32 - 1 - baseline });
            added += 1;
        }
        Ok(added)
    }

    /// Read every text line in a native-grid ink image.
    pub fn read(&self, native: &Gray) -> Recognition {
        let bm = Bitmap::binarize(native);
        let mut out = Vec::new();
        let mut worst = 1.0f32;
        for line in lines(&bm) {
            let (text, conf) = self.read_line(&bm, line);
            if !text.trim().is_empty() {
                worst = worst.min(conf);
                out.push(text);
            }
        }
        Recognition { confidence: Some(if out.is_empty() { 1.0 } else { worst }), lines: out }
    }

    fn read_line(&self, bm: &Bitmap, line: (u32, u32)) -> (String, f32) {
        let segs = segments(bm, line);
        let base = baseline(&segs);
        let mut text = String::new();
        let mut sims = Vec::new();
        let mut prev_x1: Option<u32> = None;
        for s in &segs {
            if let Some(px) = prev_x1
                && self.space > 0
                && s.x0 - px >= self.space
            {
                text.push(' ');
            }
            prev_x1 = Some(s.x1);
            let drop = s.y1 as i32 - 1 - base;
            let seg = bm.crop(s.x0, s.y0, s.x1 - s.x0, s.y1 - s.y0);
            let (best, sim) = self.best(&seg, drop);
            if sim >= ACCEPT || s.x1 - s.x0 <= self.max_w() {
                text.push(if sim >= ACCEPT { best } else { '?' });
                sims.push(sim);
            } else {
                for (ch, sim) in self.split(bm, s, base) {
                    text.push(ch);
                    sims.push(sim);
                }
            }
        }
        let conf = sims.iter().copied().fold(1.0f32, f32::min);
        (text, conf)
    }

    /// Best glyph for a tight segment bitmap.
    fn best(&self, seg: &Bitmap, drop: i32) -> (char, f32) {
        let mut best = ('?', 0.0f32);
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
                best = (g.ch, sim);
            }
        }
        best
    }

    /// Greedy left-to-right split of a segment wider than any glyph.
    fn split(&self, bm: &Bitmap, s: &Segment, base: i32) -> Vec<(char, f32)> {
        let mut out = Vec::new();
        let mut x = s.x0;
        while x < s.x1 {
            let mut best: Option<(char, f32, u32)> = None;
            for g in &self.glyphs {
                let gw = g.bitmap.w;
                if gw == 0 || x + gw > s.x1 + 1 {
                    continue;
                }
                let bottom = base + g.drop;
                let top = bottom + 1 - g.bitmap.h as i32;
                let mut cand = Bitmap { w: gw, h: g.bitmap.h, bits: Vec::with_capacity((gw * g.bitmap.h) as usize) };
                for y in top..=bottom {
                    for dx in 0..gw {
                        cand.bits.push(bm.at((x + dx) as i32, y));
                    }
                }
                let sim = iou(&g.bitmap, &cand, 0, 0);
                // Prefer wider glyphs on near-ties so "m" beats "r" + "n".
                if best.is_none_or(|(_, b, bw)| sim > b + 0.02 || (sim > b - 0.02 && gw > bw)) {
                    best = Some((g.ch, sim, gw));
                }
            }
            match best {
                Some((ch, sim, w)) if sim >= ACCEPT => {
                    out.push((ch, sim));
                    x += w;
                    while x < s.x1 && !(s.y0..s.y1).any(|y| bm.at(x as i32, y as i32)) {
                        x += 1;
                    }
                }
                _ => {
                    out.push(('?', 0.0));
                    break;
                }
            }
        }
        out
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

    /// Printable ASCII, letters first so ambiguous shapes (`l`/`I`/`|`) resolve to letters.
    const CHARS: &str =
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.,:;'\"!?-+=/\\()[]<>{}%&*#@$^_~`|";

    impl Atlas {
        /// Render a system font 1-bit with GDI, exactly as a classic Win32
        /// client draws it. `px` is the em height (LOGFONT lfHeight = -px).
        pub fn from_gdi(face: &str, px: i32, bold: bool) -> windows::core::Result<Atlas> {
            let name = format!("gdi:{face}:{px}{}", if bold { ":bold" } else { "" });
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

                for ch in CHARS.chars() {
                    std::ptr::write_bytes(bits as *mut u8, 0, (w * h * 4) as usize);
                    let s: Vec<u16> = ch.to_string().encode_utf16().collect();
                    let _ = TextOutW(dc, pad, pad, &s);
                    let _ = GdiFlush();
                    let px = std::slice::from_raw_parts(bits as *const u8, (w * h * 4) as usize);
                    let full = Bitmap {
                        w: w as u32,
                        h: h as u32,
                        bits: px.as_chunks::<4>().0.iter().map(|p| p[1] > 127).collect(),
                    };
                    if let Some((x0, y0, x1, y1)) = full.ink_bbox(0, w as u32, 0, h as u32) {
                        atlas.glyphs.push(Glyph {
                            ch,
                            bitmap: full.crop(x0, y0, x1 - x0, y1 - y0),
                            drop: y1 as i32 - 1 - baseline_row,
                        });
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
        let a = Atlas::from_gdi("Arial", 12, false).unwrap();
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
            let g = a.glyphs.iter().find(|g| g.ch == ch).unwrap();
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
