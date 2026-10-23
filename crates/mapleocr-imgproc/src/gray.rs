use crate::BgraView;

/// Single-channel f32 image (0..=255).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Gray {
    pub width: u32,
    pub height: u32,
    pub data: Vec<f32>,
}

impl Gray {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, data: vec![0.0; width as usize * height as usize] }
    }

    /// Rec. 601 luma.
    pub fn from_bgra(v: BgraView<'_>) -> Self {
        Self::from_bgra_with(v, |[b, g, r, _]| 0.114 * b as f32 + 0.587 * g as f32 + 0.299 * r as f32)
    }

    pub fn from_bgra_with(v: BgraView<'_>, f: impl Fn([u8; 4]) -> f32) -> Self {
        let mut data = Vec::with_capacity(v.width as usize * v.height as usize);
        for y in 0..v.height {
            data.extend(v.row(y).as_chunks::<4>().0.iter().map(|&px| f(px)));
        }
        Self { width: v.width, height: v.height, data }
    }

    #[inline]
    pub fn at(&self, x: u32, y: u32) -> f32 {
        self.data[y as usize * self.width as usize + x as usize]
    }

    /// Bilinear resample to `w x h` (pixel-center aligned).
    pub fn resize(&self, w: u32, h: u32) -> Gray {
        let mut out = Gray::new(w.max(1), h.max(1));
        if self.width == 0 || self.height == 0 {
            return out;
        }
        let sx = self.width as f32 / out.width as f32;
        let sy = self.height as f32 / out.height as f32;
        for y in 0..out.height {
            let fy = ((y as f32 + 0.5) * sy - 0.5).clamp(0.0, (self.height - 1) as f32);
            let y0 = fy as u32;
            let y1 = (y0 + 1).min(self.height - 1);
            let ty = fy - y0 as f32;
            for x in 0..out.width {
                let fx = ((x as f32 + 0.5) * sx - 0.5).clamp(0.0, (self.width - 1) as f32);
                let x0 = fx as u32;
                let x1 = (x0 + 1).min(self.width - 1);
                let tx = fx - x0 as f32;
                let top = self.at(x0, y0) * (1.0 - tx) + self.at(x1, y0) * tx;
                let bot = self.at(x0, y1) * (1.0 - tx) + self.at(x1, y1) * tx;
                out.data[(y * out.width + x) as usize] = top * (1.0 - ty) + bot * ty;
            }
        }
        out
    }

    /// Box-filter downscale by an integer factor (drops the remainder).
    pub fn downscale(&self, f: u32) -> Gray {
        let (w, h) = (self.width / f, self.height / f);
        let mut out = Gray::new(w, h);
        let norm = 1.0 / (f * f) as f32;
        for y in 0..h {
            for x in 0..w {
                let mut s = 0.0;
                for dy in 0..f {
                    let row = ((y * f + dy) * self.width + x * f) as usize;
                    s += self.data[row..row + f as usize].iter().sum::<f32>();
                }
                out.data[(y * w + x) as usize] = s * norm;
            }
        }
        out
    }
}
