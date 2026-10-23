//! ROI preprocessing: color keys and content-aware crops, producing dark text
//! on a light background for the recognizers.

use mapleocr_imgproc::{BgraImage, BgraView, Rect};

/// How to turn ROI pixels into "ink" (text = 255, background = 0).
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum Key {
    Luma,
    /// Pass pixels through untouched (whole-screen scans with mixed colors).
    Raw,
    /// Dark pixels are ink: dark text on light panels (dialogue, quest helper).
    Dark,
    /// Bright, unsaturated pixels only (white text with a shadow).
    White,
    /// Distance to a text color, for text over translucent/busy backgrounds.
    Color([u8; 3]),
}

impl Key {
    pub fn parse(s: Option<&str>) -> Result<Self, String> {
        match s {
            None | Some("luma") => Ok(Self::Luma),
            Some("raw") => Ok(Self::Raw),
            Some("dark") => Ok(Self::Dark),
            Some("white") => Ok(Self::White),
            Some(c) if c.len() == 7 && c.starts_with('#') => {
                let h = |i: usize| u8::from_str_radix(&c[i..i + 2], 16).map_err(|e| e.to_string());
                Ok(Self::Color([h(1)?, h(3)?, h(5)?]))
            }
            Some(other) => Err(format!("unknown key {other:?}; use luma, white or #rrggbb")),
        }
    }

    #[inline]
    pub fn ink([b, g, r, _]: [u8; 4], key: Key) -> u8 {
        match key {
            // Bright pixels are ink: light text on dark UI (tooltips, status bar).
            Key::Luma | Key::Raw => ((299 * r as u32 + 587 * g as u32 + 114 * b as u32) / 1000) as u8,
            Key::Dark => 255 - ((299 * r as u32 + 587 * g as u32 + 114 * b as u32) / 1000) as u8,
            Key::White => {
                let lo = r.min(g).min(b) as i32;
                let hi = r.max(g).max(b) as i32;
                if hi - lo >= 60 { 0 } else { ((lo - 150) * 5 / 2).clamp(0, 255) as u8 }
            }
            Key::Color([cr, cg, cb]) => {
                let d = (r as i32 - cr as i32).pow(2) + (g as i32 - cg as i32).pow(2) + (b as i32 - cb as i32).pow(2);
                (255.0 - (d as f32).sqrt() * 2.0).clamp(0.0, 255.0) as u8
            }
        }
    }
}

/// Content-aware crop modes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Crop {
    None,
    /// Keep the uniform UI panel that starts at the ROI's left edge; stop
    /// where it ends (the minimap's width varies with the map name).
    Panel,
    /// Keep only the tallest band of rows dominated by this color: a label
    /// whose position varies (the NPC name plate under the portrait).
    Band([u8; 3]),
}

impl Crop {
    pub fn parse(s: Option<&str>) -> Result<Self, String> {
        match s {
            None | Some("none") => Ok(Self::None),
            Some("panel") => Ok(Self::Panel),
            Some(b) if b.starts_with("band:") => match Key::parse(Some(&b[5..]))? {
                Key::Color(c) => Ok(Self::Band(c)),
                _ => Err(format!("crop {b:?}: expected band:#rrggbb")),
            },
            Some(other) => Err(format!("unknown crop {other:?}; use none, panel or band:#rrggbb")),
        }
    }
}

fn dist2(a: [u8; 4], b: [u8; 4]) -> i32 {
    (0..3).map(|i| (a[i] as i32 - b[i] as i32).pow(2)).sum()
}

/// Right edge (exclusive) of the panel whose color is sampled at the ROI's
/// left edge. Columns belong to the panel while a fair share of their pixels
/// match the panel color (text never covers a whole column).
pub fn panel_right_edge(img: BgraView<'_>) -> u32 {
    if img.width < 4 || img.height == 0 {
        return img.width;
    }
    // Panel color: the most common pixel among the first two columns.
    let mut samples: Vec<[u8; 4]> = (0..img.height).flat_map(|y| [img.pixel(0, y), img.pixel(1, y)]).collect();
    samples.sort_unstable();
    let mut best = (samples[0], 0usize);
    let mut run = (samples[0], 0usize);
    for &s in &samples {
        if s == run.0 {
            run.1 += 1;
        } else {
            run = (s, 1);
        }
        if run.1 > best.1 {
            best = run;
        }
    }
    let panel = best.0;
    let frac = |x: u32| {
        (0..img.height).filter(|&y| dist2(img.pixel(x, y), panel) < 30 * 30).count() as f32 / img.height as f32
    };
    const GAP: u32 = 3;
    let mut x = 2;
    while x + GAP <= img.width {
        if (x..x + GAP).all(|c| frac(c) < 0.2) {
            return x;
        }
        x += 1;
    }
    img.width
}

/// Key + crop + pad: returns an opaque BGRA image of dark text on white,
/// padded by `pad` pixels and upscaled by `upscale` (bilinear), ready for OCR.
/// Rows `y0..y1` forming the longest run where at least half the pixels are
/// near `color`.
fn band_rows(img: BgraView<'_>, [r, g, b]: [u8; 3]) -> Option<(u32, u32)> {
    let target = [b, g, r, 255];
    let dominated =
        |y: u32| (0..img.width).filter(|&x| dist2(img.pixel(x, y), target) < 45 * 45).count() * 2 >= img.width as usize;
    let mut best: Option<(u32, u32)> = None;
    let mut start = None;
    for y in 0..=img.height {
        match (y < img.height && dominated(y), start) {
            (true, None) => start = Some(y),
            (false, Some(s)) => {
                if best.is_none_or(|(a, b)| y - s > b - a) {
                    best = Some((s, y));
                }
                start = None;
            }
            _ => {}
        }
    }
    best
}

/// The part of the ROI the recognizer should see, in ROI pixels.
pub fn crop_rect(img: BgraView<'_>, crop: Crop) -> Rect {
    match crop {
        Crop::None => Rect::new(0, 0, img.width, img.height),
        Crop::Panel => Rect::new(0, 0, panel_right_edge(img), img.height),
        Crop::Band(c) => match band_rows(img, c) {
            Some((y0, y1)) => Rect::new(0, y0 as i32, img.width, y1 - y0),
            None => Rect::new(0, 0, 0, 0),
        },
    }
}

pub fn prepare(img: BgraView<'_>, key: Key, crop: Crop, pad: u32, upscale: f32) -> BgraImage {
    let img = img.crop(crop_rect(img, crop));
    if key == Key::Raw {
        let (ow, oh) = (img.width + 2 * pad, img.height + 2 * pad);
        let mut out = BgraImage::new(ow, oh);
        out.data.fill(255);
        for y in 0..img.height {
            let o = (((y + pad) * ow + pad) * 4) as usize;
            out.data[o..o + img.width as usize * 4].copy_from_slice(img.row(y));
        }
        return out;
    }
    let mut ink = mapleocr_imgproc::Gray::from_bgra_with(img, |px| Key::ink(px, key) as f32);
    if (upscale - 1.0).abs() > 1e-3 {
        ink = ink.resize((ink.width as f32 * upscale).round() as u32, (ink.height as f32 * upscale).round() as u32);
    }
    let (ow, oh) = (ink.width + 2 * pad, ink.height + 2 * pad);
    let mut out = BgraImage::new(ow, oh);
    out.data.fill(255);
    for y in 0..ink.height {
        for x in 0..ink.width {
            let v = 255 - ink.at(x, y).round().clamp(0.0, 255.0) as u8;
            let o = (((y + pad) * ow + x + pad) * 4) as usize;
            out.data[o..o + 3].fill(v);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn key_parsing_and_ink() {
        assert_eq!(Key::parse(Some("#ffff00")), Ok(Key::Color([255, 255, 0])));
        assert!(Key::parse(Some("blue")).is_err());
        assert_eq!(Key::ink([255, 255, 255, 255], Key::White), 255);
        assert_eq!(Key::ink([206, 189, 151, 255], Key::White), 2); // minimap panel stays background
        assert_eq!(Key::ink([0, 255, 255, 255], Key::Color([255, 255, 0])), 255);
    }

    #[test]
    fn panel_edge_stops_where_panel_color_ends() {
        let (w, h) = (60u32, 20u32);
        let mut img = BgraImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let px = if x >= 40 {
                    [10, 80, 30, 255] // world
                } else if (10..14).contains(&x) && y % 2 == 0 {
                    [255, 255, 255, 255] // text strokes
                } else {
                    [206, 189, 151, 255] // panel
                };
                let o = ((y * w + x) * 4) as usize;
                img.data[o..o + 4].copy_from_slice(&px);
            }
        }
        assert_eq!(panel_right_edge(img.view()), 40);
        let out = prepare(img.view(), Key::White, Crop::Panel, 4, 1.0);
        assert_eq!((out.width, out.height), (48, 28));
    }
}
