//! Image primitives shared by the mpl-ocr crates: BGRA views, grayscale
//! images, resampling, normalized cross-correlation search, and ROI hashing.
//!
//! Pixel format throughout is BGRA8 (what D3D11/WGC deliver).

mod gray;
mod io;
mod native;
mod search;

pub use gray::Gray;
pub use io::{ImageError, load_png, save_png};
pub use native::{to_native, to_native_aligned};
pub use search::{Match, ncc_search};

/// Axis-aligned rectangle in pixels.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub struct Rect {
    pub x: i32,
    pub y: i32,
    pub w: u32,
    pub h: u32,
}

impl Rect {
    pub const fn new(x: i32, y: i32, w: u32, h: u32) -> Self {
        Self { x, y, w, h }
    }

    pub fn right(&self) -> i32 {
        self.x + self.w as i32
    }

    pub fn bottom(&self) -> i32 {
        self.y + self.h as i32
    }

    /// Intersection with `0..w, 0..h`.
    pub fn clamp_to(&self, w: u32, h: u32) -> Rect {
        let x0 = self.x.clamp(0, w as i32);
        let y0 = self.y.clamp(0, h as i32);
        let x1 = self.right().clamp(0, w as i32);
        let y1 = self.bottom().clamp(0, h as i32);
        Rect::new(x0, y0, (x1 - x0).max(0) as u32, (y1 - y0).max(0) as u32)
    }
}

/// 64-bit xxh3 of ROI pixels, used for dirty detection and result caching.
#[inline]
pub fn hash_pixels(data: &[u8]) -> u64 {
    xxhash_rust::xxh3::xxh3_64(data)
}

/// Borrowed BGRA8 image with an arbitrary row stride (in bytes).
#[derive(Clone, Copy, Debug)]
pub struct BgraView<'a> {
    pub width: u32,
    pub height: u32,
    pub stride: usize,
    pub data: &'a [u8],
}

impl<'a> BgraView<'a> {
    /// Tightly packed view (`stride = width * 4`).
    pub fn packed(width: u32, height: u32, data: &'a [u8]) -> Self {
        assert!(data.len() >= width as usize * height as usize * 4, "BGRA buffer too small");
        Self { width, height, stride: width as usize * 4, data }
    }

    #[inline]
    pub fn pixel(&self, x: u32, y: u32) -> [u8; 4] {
        let o = y as usize * self.stride + x as usize * 4;
        [self.data[o], self.data[o + 1], self.data[o + 2], self.data[o + 3]]
    }

    #[inline]
    pub fn row(&self, y: u32) -> &'a [u8] {
        let o = y as usize * self.stride;
        &self.data[o..o + self.width as usize * 4]
    }

    /// Sub-view, clamped to the image.
    pub fn crop(&self, r: Rect) -> BgraView<'a> {
        let c = r.clamp_to(self.width, self.height);
        let start = c.y as usize * self.stride + c.x as usize * 4;
        let end = if c.h == 0 { start } else { start + (c.h as usize - 1) * self.stride + c.w as usize * 4 };
        BgraView { width: c.w, height: c.h, stride: self.stride, data: &self.data[start..end] }
    }

    pub fn to_owned(&self) -> BgraImage {
        let mut data = Vec::with_capacity(self.width as usize * self.height as usize * 4);
        for y in 0..self.height {
            data.extend_from_slice(self.row(y));
        }
        BgraImage { width: self.width, height: self.height, data }
    }
}

/// Owned, tightly packed BGRA8 image.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BgraImage {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
}

impl BgraImage {
    pub fn new(width: u32, height: u32) -> Self {
        Self { width, height, data: vec![0; width as usize * height as usize * 4] }
    }

    pub fn view(&self) -> BgraView<'_> {
        BgraView::packed(self.width, self.height, &self.data)
    }
}
