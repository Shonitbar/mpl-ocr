use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use anyhow::Result;
use mapleocr_capture::RoiImage;

/// Write a BGRA ROI as an RGBA PNG.
pub fn write_png(path: &Path, img: &RoiImage) -> Result<()> {
    let mut rgba = img.data().to_vec();
    for px in rgba.as_chunks_mut::<4>().0 {
        px.swap(0, 2);
        px[3] = 255; // captured alpha is meaningless for an opaque window
    }
    let mut enc = png::Encoder::new(BufWriter::new(File::create(path)?), img.width(), img.height());
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(&rgba)?;
    Ok(())
}
