//! Undo the client's UI upscale: area-resample a client-space crop back onto
//! the native UI grid, where 1-bit fonts and sprites are pixel-exact again.
//!
//! The client draws its UI on a canvas `canvas_height` lines tall and stretches
//! it by `scale`, so native pixel `n` covers client pixels `[n*s, (n+1)*s)`.

use crate::Gray;

/// For each output index, the contributing input indices and weights.
fn taps(origin: f64, len: u32, scale: f32) -> (i32, Vec<Vec<(u32, f32)>>) {
    let s = scale as f64;
    let (a, b) = (origin, origin + len as f64);
    let n0 = (a / s - 1e-6).ceil() as i32;
    let n1 = (b / s + 1e-6).floor() as i32;
    let mut out = Vec::with_capacity((n1 - n0).max(0) as usize);
    for n in n0..n1 {
        let (lo, hi) = (n as f64 * s - a, (n + 1) as f64 * s - a);
        let mut t = Vec::with_capacity(3);
        let mut c = lo.floor().max(0.0) as u32;
        while (c as f64) < hi && c < len {
            let w = (hi.min(c as f64 + 1.0) - lo.max(c as f64)) as f32;
            if w > 1e-4 {
                t.push((c, w / scale));
            }
            c += 1;
        }
        out.push(t);
    }
    (n0, out)
}

/// Area-resample `img` (whose top-left sits at client pixel `origin`) onto the
/// native grid. Returns the native image and the native coordinates of its
/// top-left. Only native pixels fully inside `img` are produced.
pub fn to_native(img: &Gray, origin: (i32, i32), scale: f32) -> (Gray, (i32, i32)) {
    resample(img, (origin.0 as f64, origin.1 as f64), scale)
}

/// Like [`to_native`], but searches a sub-pixel grid phase (quarters of a
/// native pixel on each axis) and keeps the crispest result (highest
/// variance). Absorbs small errors in the assumed canvas size, which otherwise
/// put 1-px strokes between native pixels.
pub fn to_native_aligned(img: &Gray, origin: (i32, i32), scale: f32) -> Gray {
    to_native_phase(img, origin, scale, best_phase(img, origin, scale))
}

/// The sub-pixel grid phase (quarters of a native pixel per axis) that makes
/// `img` crispest. Constant for a fixed ROI and window size, so callers can
/// compute it once and reuse it with [`to_native_phase`].
pub fn best_phase(img: &Gray, origin: (i32, i32), scale: f32) -> (f64, f64) {
    const PHASES: [f64; 4] = [0.0, 0.25, 0.5, 0.75];
    let mut best = (f32::MIN, (0.0, 0.0));
    for py in PHASES {
        for px in PHASES {
            let g = to_native_phase(img, origin, scale, (px, py));
            let n = g.data.len().max(1) as f32;
            let mean = g.data.iter().sum::<f32>() / n;
            let var = g.data.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n;
            if var > best.0 {
                best = (var, (px, py));
            }
        }
    }
    best.1
}

/// [`to_native`] with the native grid shifted by `phase` native pixels.
pub fn to_native_phase(img: &Gray, origin: (i32, i32), scale: f32, phase: (f64, f64)) -> Gray {
    let o = (origin.0 as f64 - phase.0 * scale as f64, origin.1 as f64 - phase.1 * scale as f64);
    resample(img, o, scale).0
}

fn resample(img: &Gray, origin: (f64, f64), scale: f32) -> (Gray, (i32, i32)) {
    let (nx0, xt) = taps(origin.0, img.width, scale);
    let (ny0, yt) = taps(origin.1, img.height, scale);
    let mut out = Gray::new(xt.len() as u32, yt.len() as u32);
    let iw = img.width as usize;
    // Horizontal pass into a scratch buffer (rows x native columns), then vertical.
    let mut horiz = vec![0.0f32; img.height as usize * xt.len()];
    for y in 0..img.height as usize {
        let row = &img.data[y * iw..(y + 1) * iw];
        for (nx, t) in xt.iter().enumerate() {
            horiz[y * xt.len() + nx] = t.iter().map(|&(c, w)| row[c as usize] * w).sum();
        }
    }
    for (ny, t) in yt.iter().enumerate() {
        for nx in 0..xt.len() {
            out.data[ny * xt.len() + nx] = t.iter().map(|&(r, w)| horiz[r as usize * xt.len() + nx] * w).sum();
        }
    }
    (out, (nx0, ny0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integer_scale_is_exact_box_filter() {
        // 2x upscale of a 3x2 native image, cropped at client (2, 0).
        let native = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0];
        let mut client = Gray::new(6, 4);
        for y in 0..4 {
            for x in 0..6 {
                client.data[y * 6 + x] = native[(y / 2) * 3 + x / 2];
            }
        }
        let (n, o) = to_native(&client, (0, 0), 2.0);
        assert_eq!((n.width, n.height, o), (3, 2, (0, 0)));
        assert_eq!(n.data, native);
        // Crop starting mid-pixel: only whole native pixels come out.
        let crop = Gray {
            width: 5,
            height: 4,
            data: (0..4).flat_map(|y| client.data[y * 6 + 1..y * 6 + 6].to_vec()).collect(),
        };
        let (n, o) = to_native(&crop, (1, 0), 2.0);
        assert_eq!(o, (1, 0));
        assert_eq!(n.data, vec![20.0, 30.0, 50.0, 60.0]);
    }

    #[test]
    fn fractional_scale_preserves_flat_regions() {
        let client = Gray { width: 25, height: 25, data: vec![100.0; 625] };
        let (n, _) = to_native(&client, (0, 0), 1.5625);
        assert_eq!((n.width, n.height), (16, 16));
        assert!(n.data.iter().all(|&v| (v - 100.0).abs() < 1e-3));
    }
}
