//! Normalized cross-correlation template search, coarse to fine.
//!
//! Runs at calibration time only (startup, resize, anchor loss), never per
//! frame, so it favours simplicity over SIMD.

use crate::{Gray, Rect};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Match {
    /// Top-left of the best match in image pixels.
    pub x: i32,
    pub y: i32,
    /// Normalized cross-correlation in -1..=1.
    pub score: f32,
}

/// Window sums over `w x h` windows from an integral image.
struct Integral {
    w: usize,
    s1: Vec<f64>,
    s2: Vec<f64>,
}

impl Integral {
    fn new(img: &Gray) -> Self {
        let (w, h) = (img.width as usize + 1, img.height as usize + 1);
        let mut s1 = vec![0.0; w * h];
        let mut s2 = vec![0.0; w * h];
        for y in 1..h {
            let (mut r1, mut r2) = (0.0, 0.0);
            for x in 1..w {
                let v = img.data[(y - 1) * (w - 1) + (x - 1)] as f64;
                r1 += v;
                r2 += v * v;
                s1[y * w + x] = s1[(y - 1) * w + x] + r1;
                s2[y * w + x] = s2[(y - 1) * w + x] + r2;
            }
        }
        Self { w, s1, s2 }
    }

    fn window(&self, x: usize, y: usize, tw: usize, th: usize) -> (f64, f64) {
        let w = self.w;
        let f = |s: &[f64]| s[(y + th) * w + x + tw] - s[y * w + x + tw] - s[(y + th) * w + x] + s[y * w + x];
        (f(&self.s1), f(&self.s2))
    }
}

/// Exhaustive NCC over top-left positions in `area` (clamped so the template fits).
fn search_exhaustive(img: &Gray, tmpl: &Gray, area: Rect) -> Option<Match> {
    let (tw, th) = (tmpl.width as usize, tmpl.height as usize);
    if tw == 0 || th == 0 || tw > img.width as usize || th > img.height as usize {
        return None;
    }
    let n = (tw * th) as f64;
    let tmean = tmpl.data.iter().map(|&v| v as f64).sum::<f64>() / n;
    let tz: Vec<f32> = tmpl.data.iter().map(|&v| (v as f64 - tmean) as f32).collect();
    let tnorm = tz.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
    if tnorm < 1e-6 {
        return None;
    }
    let integral = Integral::new(img);
    let x_max = (img.width as usize - tw) as i32;
    let y_max = (img.height as usize - th) as i32;
    let (x0, y0) = (area.x.clamp(0, x_max), area.y.clamp(0, y_max));
    let (x1, y1) = (area.right().clamp(0, x_max), area.bottom().clamp(0, y_max));
    let iw = img.width as usize;
    let mut best: Option<Match> = None;
    for y in y0..=y1 {
        for x in x0..=x1 {
            let (s1, s2) = integral.window(x as usize, y as usize, tw, th);
            let var = s2 - s1 * s1 / n;
            if var < 1e-6 {
                continue;
            }
            let mut num = 0.0f32;
            for ty in 0..th {
                let row = &img.data[(y as usize + ty) * iw + x as usize..][..tw];
                num += row.iter().zip(&tz[ty * tw..(ty + 1) * tw]).map(|(a, b)| a * b).sum::<f32>();
            }
            let score = (num as f64 / (var.sqrt() * tnorm)) as f32;
            if best.is_none_or(|b| score > b.score) {
                best = Some(Match { x, y, score });
            }
        }
    }
    best
}

/// The `k` best non-overlapping matches (exhaustive; use on small images).
pub fn ncc_peaks(img: &Gray, tmpl: &Gray, k: usize) -> Vec<Match> {
    let (tw, th) = (tmpl.width as usize, tmpl.height as usize);
    if tw == 0 || th == 0 || tw > img.width as usize || th > img.height as usize {
        return Vec::new();
    }
    let n = (tw * th) as f64;
    let tmean = tmpl.data.iter().map(|&v| v as f64).sum::<f64>() / n;
    let tz: Vec<f32> = tmpl.data.iter().map(|&v| (v as f64 - tmean) as f32).collect();
    let tnorm = tz.iter().map(|&v| (v as f64) * (v as f64)).sum::<f64>().sqrt();
    if tnorm < 1e-6 {
        return Vec::new();
    }
    let integral = Integral::new(img);
    let iw = img.width as usize;
    let (xs, ys) = (iw - tw + 1, img.height as usize - th + 1);
    let mut scores = vec![f32::MIN; xs * ys];
    for y in 0..ys {
        for x in 0..xs {
            let (s1, s2) = integral.window(x, y, tw, th);
            let var = s2 - s1 * s1 / n;
            if var < 1e-6 {
                continue;
            }
            let mut num = 0.0f32;
            for ty in 0..th {
                let row = &img.data[(y + ty) * iw + x..][..tw];
                num += row.iter().zip(&tz[ty * tw..(ty + 1) * tw]).map(|(a, b)| a * b).sum::<f32>();
            }
            scores[y * xs + x] = (num as f64 / (var.sqrt() * tnorm)) as f32;
        }
    }
    let mut out: Vec<Match> = Vec::new();
    for _ in 0..k {
        let Some((i, &s)) = scores.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)) else { break };
        if s <= 0.0 {
            break;
        }
        let (x, y) = ((i % xs) as i32, (i / xs) as i32);
        out.push(Match { x, y, score: s });
        for yy in (y - th as i32).max(0)..(y + th as i32).min(ys as i32) {
            for xx in (x - tw as i32).max(0)..(x + tw as i32).min(xs as i32) {
                scores[yy as usize * xs + xx as usize] = f32::MIN;
            }
        }
    }
    out
}

/// Find `tmpl` in `img` with its top-left inside `area`. Searches a 4x
/// downscaled pyramid level first when the area is large, then refines at
/// full resolution around the coarse hit.
pub fn ncc_search(img: &Gray, tmpl: &Gray, area: Rect) -> Option<Match> {
    const F: u32 = 4;
    let large = area.w * area.h > 64 * 64 && tmpl.width >= 8 * F && tmpl.height >= 3 * F;
    if !large {
        return search_exhaustive(img, tmpl, area);
    }
    let (small_img, small_tmpl) = (img.downscale(F), tmpl.downscale(F));
    let coarse_area = Rect::new(area.x / F as i32, area.y / F as i32, area.w.div_ceil(F) + 1, area.h.div_ceil(F) + 1);
    let coarse = search_exhaustive(&small_img, &small_tmpl, coarse_area)?;
    let r = 2 * F as i32;
    let fine = Rect::new(coarse.x * F as i32 - r, coarse.y * F as i32 - r, 2 * r as u32, 2 * r as u32);
    search_exhaustive(img, tmpl, fine)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(w: u32, h: u32, seed: u32) -> Gray {
        let mut g = Gray::new(w, h);
        let mut s = seed;
        for v in &mut g.data {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            *v = (s % 256) as f32;
        }
        g
    }

    fn crop(g: &Gray, r: Rect) -> Gray {
        let mut out = Gray::new(r.w, r.h);
        for y in 0..r.h {
            for x in 0..r.w {
                out.data[(y * r.w + x) as usize] = g.at(r.x as u32 + x, r.y as u32 + y);
            }
        }
        out
    }

    #[test]
    fn finds_exact_template_small_and_pyramid() {
        let img = noise(400, 300, 7);
        // Smooth it a little so the 4x pyramid level keeps structure.
        let img = img.resize(200, 150).resize(400, 300);
        let t = crop(&img, Rect::new(271, 133, 64, 24));
        let m = ncc_search(&img, &t, Rect::new(0, 0, 400, 300)).unwrap();
        assert_eq!((m.x, m.y), (271, 133));
        assert!(m.score > 0.999);
        let m = ncc_search(&img, &t, Rect::new(265, 128, 12, 12)).unwrap();
        assert_eq!((m.x, m.y), (271, 133));
    }

    #[test]
    fn rect_clamp() {
        assert_eq!(Rect::new(-5, 10, 20, 100).clamp_to(10, 50), Rect::new(0, 10, 10, 40));
        assert_eq!(Rect::new(20, 0, 5, 5).clamp_to(10, 10).w, 0);
    }
}
