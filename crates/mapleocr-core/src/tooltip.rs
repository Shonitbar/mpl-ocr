//! Tooltips (item and skill): found by the light-blue bullet before the title,
//! then read on the native UI grid with the glyph matcher.
//!
//! The bullet is an opaque sprite of one exact color, so unlike the
//! translucent navy box it does not depend on what is behind the tooltip. All
//! layout offsets are measured from it.

use mapleocr_imgproc::{BgraView, Gray, Rect, best_phase, to_native_phase};
use mapleocr_recog::glyph::Bitmap;
use mapleocr_recog::{Atlas, Tooltip};

/// Bullet color: R 70-100, G 190-230, B >= 245.
#[inline]
fn is_bullet_color([b, g, r, _]: [u8; 4]) -> bool {
    (70..=100).contains(&r) && (190..=230).contains(&g) && b >= 245
}

/// Top-left client pixel of every title bullet in the frame.
pub fn find_bullets(img: BgraView<'_>, scale: f32) -> Vec<(i32, i32)> {
    let mut pts: Vec<(i32, i32)> = Vec::new();
    for y in 0..img.height {
        for (x, &px) in img.row(y).as_chunks::<4>().0.iter().enumerate() {
            if is_bullet_color(px) {
                pts.push((x as i32, y as i32));
            }
        }
    }
    // Cluster by proximity, keep bullet-sized round-ish blobs.
    let mut used = vec![false; pts.len()];
    let mut out = Vec::new();
    let (lo, hi) = ((4.0 * scale) as i32, (6.0 * scale).ceil() as i32);
    for i in 0..pts.len() {
        if used[i] {
            continue;
        }
        let mut cluster = vec![i];
        used[i] = true;
        let mut k = 0;
        while k < cluster.len() {
            let (cx, cy) = pts[cluster[k]];
            for j in 0..pts.len() {
                if !used[j] && (pts[j].0 - cx).abs() <= 1 && (pts[j].1 - cy).abs() <= 1 {
                    used[j] = true;
                    cluster.push(j);
                }
            }
            k += 1;
        }
        let (x0, x1) = cluster.iter().fold((i32::MAX, i32::MIN), |a, &j| (a.0.min(pts[j].0), a.1.max(pts[j].0)));
        let (y0, y1) = cluster.iter().fold((i32::MAX, i32::MIN), |a, &j| (a.0.min(pts[j].1), a.1.max(pts[j].1)));
        let (w, h) = (x1 - x0 + 1, y1 - y0 + 1);
        let n = cluster.len() as i32;
        if (lo..=hi + 2).contains(&w) && (lo..=hi + 2).contains(&h) && n * 2 >= w * h {
            out.push((x0, y0));
        }
    }
    out
}

/// A tooltip's rectangle (client pixels) and the native-grid ink of its interior.
pub struct Located {
    pub bullet: (i32, i32),
    pub rect: Rect,
}

/// Dark shadow line at the right and bottom edges of the box.
#[inline]
fn shadow([b, g, r, _]: [u8; 4]) -> bool {
    // Dark navy (R~14 G~18 B~85). The black border of the item icon must not match.
    let luma = (299 * r as i32 + 587 * g as i32 + 114 * b as i32) / 1000;
    luma < 45 && b >= 60 && b as i32 > r as i32 + 40
}

/// Box around a bullet: the corner sits at a fixed offset from the bullet, and
/// the far edges are found by the dark shadow line.
pub fn locate(img: BgraView<'_>, bullet: (i32, i32), scale: f32) -> Located {
    let (w, h) = (img.width as i32, img.height as i32);
    let left = (bullet.0 as f32 - 12.0 * scale / 1.5625).round() as i32;
    let top = (bullet.1 as f32 - 22.0 * scale / 1.5625).round() as i32;
    let max_w = (520.0 * scale) as i32;
    let max_h = (420.0 * scale) as i32;
    // Right edge: a column where the shadow line covers most of the first rows.
    let probe = (60.0 * scale) as i32;
    let mut right = left + (100.0 * scale) as i32;
    for x in left + (60.0 * scale) as i32..(left + max_w).min(w) {
        let n = (top + 6..(top + probe).min(h)).filter(|&y| shadow(img.pixel(x as u32, y as u32))).count() as i32;
        if n * 10 >= (probe - 6) * 7 {
            right = x;
            break;
        }
    }
    let mut bottom = top + (40.0 * scale) as i32;
    for y in top + (40.0 * scale) as i32..(top + max_h).min(h) {
        let n = (left + 6..(left + probe).min(w)).filter(|&x| shadow(img.pixel(x as u32, y as u32))).count() as i32;
        if n * 10 >= (probe - 6) * 7 {
            bottom = y;
            break;
        }
    }
    Located { bullet, rect: Rect::new(left, top, (right - left + 1).max(1) as u32, (bottom - top + 1).max(1) as u32) }
}

/// Native-grid ink of `rect` (bright text = ink), grid-aligned to the bullet.
pub fn native_ink(img: BgraView<'_>, rect: Rect, bullet: (i32, i32), scale: f32) -> (Gray, (i32, i32)) {
    let crop = img.crop(rect);
    let ink = Gray::from_bgra_with(crop, |[b, g, r, _]| {
        // Text is white or light gray; the fill is navy: use min channel so navy stays dark.
        r.min(g).min(b) as f32
    });
    let s = scale as f64;
    let ph = |b: i32, o: i32| (-((b - o) as f64) / s).rem_euclid(1.0);
    let phase = (ph(bullet.0, rect.x), ph(bullet.1, rect.y));
    let _ = best_phase;
    (to_native_phase(&ink, (rect.x, rect.y), scale, phase), (rect.x, rect.y))
}

/// ASCII picture of binarized native ink, for tuning.
pub fn ascii(g: &Gray) -> String {
    let bm = Bitmap::binarize(g);
    bm.bits
        .chunks(bm.w.max(1) as usize)
        .map(|r| r.iter().map(|&b| if b { '#' } else { '.' }).collect::<String>())
        .collect::<Vec<_>>()
        .join("\n")
}

/// Corner of the box: a fixed offset (7.68, 14.08 native) up-left of the bullet.
pub fn corner(bullet: (i32, i32), scale: f32) -> (i32, i32) {
    ((bullet.0 as f32 - 7.68 * scale).round() as i32, (bullet.1 as f32 - 14.08 * scale).round() as i32)
}

/// Widest and tallest tooltip we try to read, in native pixels.
const MAX_W: f32 = 460.0;
const MAX_H: f32 = 300.0;
/// The item/skill icon frame, relative to the corner, in native pixels (x0, y0, x1, y1).
const ICON: (f32, f32, f32, f32) = (7.0, 29.0, 79.0, 99.0);

/// Remembers what each tooltip said, so one that stays up while the cursor rests on an
/// item is read once, not on every scout. Keyed by its corner and a hash of its pure-white
/// pixels (text), which is unaffected by the world moving behind the translucent box.
#[derive(Default)]
pub struct Cache {
    entries: Vec<CacheEntry>,
}

/// Corner, hash of the text pixels, and the reading (name, lines).
type CacheEntry = ((i32, i32), u64, (String, Vec<String>));

/// Read every tooltip on screen. `atlas` should hold Arial 12 regular and bold plus the
/// title font.
pub fn read_all(img: BgraView<'_>, atlas: &Atlas, scale: f32) -> Vec<Tooltip> {
    read_all_cached(img, atlas, scale, &mut Cache::default())
}

pub fn read_all_cached(img: BgraView<'_>, atlas: &Atlas, scale: f32, cache: &mut Cache) -> Vec<Tooltip> {
    let mut fresh: Vec<CacheEntry> = Vec::new();
    let bullets = find_bullets(img, scale);
    let corners: Vec<(i32, i32)> = bullets.iter().map(|&b| corner(b, scale)).collect();
    let mut out = Vec::new();
    for (i, &(cx, cy)) in corners.iter().enumerate() {
        // Right limit: the next tooltip to the right on the same rows, else the widest we read.
        let mut right = cx + (MAX_W * scale) as i32;
        for (j, &(ox, oy)) in corners.iter().enumerate() {
            if j != i && ox > cx + (40.0 * scale) as i32 && (oy - cy).abs() < (120.0 * scale) as i32 {
                right = right.min(ox - 2);
            }
        }
        // The white separator under the header ends 13 px (at x1.56) before the box edge,
        // for tooltips that have one. It gives the exact width even for the last tooltip.
        if let Some(sep_end) = separator_end(img, (cx, cy), scale) {
            right = right.min(sep_end + (8.3 * scale).round() as i32);
        }
        let bottom = cy + (MAX_H * scale) as i32;
        let rect =
            Rect::new(cx, cy, (right - cx).max(1) as u32, (bottom - cy).max(1) as u32).clamp_to(img.width, img.height);
        if rect.w < 20 || rect.h < 20 {
            continue;
        }
        let key = text_hash(img, rect);
        let read = match cache.entries.iter().find(|e| e.0 == (cx, cy) && e.1 == key) {
            Some(hit) => hit.2.clone(),
            None => read_rect(img, atlas, rect, scale),
        };
        fresh.push(((cx, cy), key, read.clone()));
        out.push(Tooltip { name: read.0, lines: read.1, rect });
    }
    cache.entries = fresh;
    out
}

/// End of the separator line: a long run of near-white pixels that starts a few pixels
/// inside the left edge, somewhere below the title.
fn separator_end(img: BgraView<'_>, (cx, cy): (i32, i32), scale: f32) -> Option<i32> {
    let min_run = (65.0 * scale) as i32;
    let y_end = (cy + (MAX_H * 1.4 * scale) as i32).min(img.height as i32);
    for y in cy + (30.0 * scale) as i32..y_end {
        let white = |x: i32| {
            x < img.width as i32 && {
                let [b, g, r, _] = img.pixel(x as u32, y as u32);
                r.min(g).min(b) >= 232
            }
        };
        let Some(x0) = (cx..cx + 24).find(|&x| white(x)) else { continue };
        let mut x = x0;
        while white(x) {
            x += 1;
        }
        // A real separator has tooltip fill (navy) just above and below it; a white input
        // box or window frame does not.
        let navy = |yy: i32| {
            let [b, g, r, _] = img.pixel((x0 + 12) as u32, yy.clamp(0, img.height as i32 - 1) as u32);
            b as i32 > r as i32 + 15 && b as i32 > g as i32 + 8
        };
        if x - x0 >= min_run && navy(y - 4) && navy(y + 4) {
            return Some(x);
        }
    }
    None
}

/// How ink is extracted from screen pixels: a ramp on the minimum channel (below `lo`
/// nothing, at 255 full ink), then a binarization of the resampled result.
#[derive(Clone, Copy)]
enum Binarize {
    Fixed(f32),
    Adaptive,
    Banded,
}

const VARIANTS: [(f32, Binarize); 6] = [
    (140.0, Binarize::Fixed(65.0)),
    (140.0, Binarize::Fixed(90.0)),
    (215.0, Binarize::Fixed(50.0)),
    (0.0, Binarize::Adaptive),
    (0.0, Binarize::Banded),
    (100.0, Binarize::Fixed(70.0)),
];

/// Binarized native-grid images of a tooltip rect under several ink extractions. Which
/// one reads a line best depends on what is behind the translucent box, so all are tried.
pub fn prepare_variants(img: BgraView<'_>, rect: Rect, scale: f32) -> Vec<Bitmap> {
    let crop = img.crop(rect);
    let ink_at = |lo: f32| -> Gray {
        let mut ink = Gray::from_bgra_with(crop, |[b, g, r, _]| {
            let m = r.min(g).min(b) as f32;
            if lo <= 0.0 { m } else { ((m - lo) / (255.0 - lo)).clamp(0.0, 1.0) * 255.0 }
        });
        // Erase the icon frame: its rows share lines with the text column.
        let (ix0, iy0) = (((ICON.0 * scale) as i32).max(0), ((ICON.1 * scale) as i32).max(0));
        let (ix1, iy1) = ((ICON.2 * scale) as i32, (ICON.3 * scale) as i32);
        for y in iy0.min(ink.height as i32)..iy1.min(ink.height as i32) {
            for x in ix0.min(ink.width as i32)..ix1.min(ink.width as i32) {
                ink.data[(y as u32 * ink.width + x as u32) as usize] = 0.0;
            }
        }
        ink
    };
    // One grid phase for all variants, from the ramped ink.
    let phase = best_phase(&ink_at(140.0), (rect.x, rect.y), scale);
    VARIANTS
        .iter()
        .map(|&(lo, mode)| {
            let native = to_native_phase(&ink_at(lo), (rect.x, rect.y), scale, phase);
            match mode {
                Binarize::Fixed(t) => Bitmap::from_threshold(&native, t),
                Binarize::Adaptive => Bitmap::binarize(&native),
                Binarize::Banded => Bitmap::binarize_banded(&native, 120.0),
            }
        })
        .collect()
}

/// Hash of the pixels that are pure white (text, separators): stable under whatever shows
/// through the translucent box.
fn text_hash(img: BgraView<'_>, rect: Rect) -> u64 {
    let crop = img.crop(rect);
    let mut bits = Vec::with_capacity((crop.width * crop.height) as usize / 8 + 1);
    let mut acc = 0u8;
    let mut n = 0;
    for y in 0..crop.height {
        for &[b, g, r, _] in crop.row(y).as_chunks::<4>().0 {
            acc = (acc << 1) | u8::from(r.min(g).min(b) >= 240);
            n += 1;
            if n == 8 {
                bits.push(acc);
                (acc, n) = (0, 0);
            }
        }
    }
    bits.push(acc);
    mapleocr_imgproc::hash_pixels(&bits)
}

/// Read a tooltip rect: every variant reads every line, and per row band the reading
/// the font fits best wins.
pub fn read_rect(img: BgraView<'_>, atlas: &Atlas, rect: Rect, scale: f32) -> (String, Vec<String>) {
    let t0 = std::time::Instant::now();
    let variants = prepare_variants(img, rect, scale);
    let t1 = t0.elapsed();
    let mut candidates: Vec<mapleocr_recog::glyph::ScoredLine> =
        variants.into_iter().flat_map(|bm| atlas.read_scored(bm)).collect();
    if std::env::var_os("MAPLEOCR_PROFILE").is_some() {
        eprintln!(
            "tooltip {}x{}: prepare {:?}, read {:?}, {} glyphs",
            rect.w,
            rect.h,
            t1,
            t0.elapsed() - t1,
            atlas.glyphs.len()
        );
    }
    candidates.sort_by(|a, b| b.quality.total_cmp(&a.quality));
    let mut accepted: Vec<mapleocr_recog::glyph::ScoredLine> = Vec::new();
    for c in candidates {
        let clashes = accepted.iter().any(|a| {
            let y_overlap = c.rows.1.min(a.rows.1) as i32 - c.rows.0.max(a.rows.0) as i32;
            let x_overlap = c.x.1.min(a.x.1) as i32 - c.x.0.max(a.x.0) as i32;
            let min_h = (c.rows.1 - c.rows.0).min(a.rows.1 - a.rows.0) as i32;
            y_overlap * 2 > min_h && x_overlap > 0
        });
        if !clashes && c.quality > 1.0 {
            accepted.push(c);
        }
    }
    accepted.sort_by_key(|l| (l.rows.0, l.x.0));
    let mut lines = accepted.into_iter().map(|l| clean(&l.text)).filter(|l| !l.is_empty());
    let name = lines.next().unwrap_or_default();
    (name, lines.collect())
}

/// Drop unknown-glyph junk at the ends of a line (world pixels past the box edge, cursor).
fn clean(line: &str) -> String {
    line.trim_matches(|c: char| c == '?' || c.is_whitespace()).to_owned()
}

#[cfg(test)]
mod tests {
    use super::*;
    use mapleocr_imgproc::BgraImage;

    fn put(img: &mut BgraImage, x: u32, y: u32, w: u32, h: u32, [r, g, b]: [u8; 3]) {
        for yy in y..y + h {
            for xx in x..x + w {
                let o = ((yy * img.width + xx) * 4) as usize;
                img.data[o..o + 4].copy_from_slice(&[b, g, r, 255]);
            }
        }
    }

    #[test]
    fn finds_title_bullets_and_nothing_else() {
        let mut img = BgraImage::new(400, 300);
        put(&mut img, 0, 0, 400, 300, [20, 79, 220]); // blue sky
        put(&mut img, 100, 80, 7, 7, [85, 208, 255]); // a bullet
        put(&mut img, 250, 200, 7, 6, [86, 210, 255]);
        put(&mut img, 10, 10, 30, 1, [85, 208, 255]); // a line of the same color: not a bullet
        put(&mut img, 300, 30, 14, 9, [85, 208, 255]); // too big
        let mut b = find_bullets(img.view(), 1.5625);
        b.sort();
        assert_eq!(b, vec![(100, 80), (250, 200)]);
        assert_eq!(corner((100, 80), 1.5625), (88, 58));
    }
}
