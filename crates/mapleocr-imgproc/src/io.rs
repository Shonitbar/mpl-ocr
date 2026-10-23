use std::fs::File;
use std::io::{BufReader, BufWriter};
use std::path::Path;

use crate::{BgraImage, BgraView};

#[derive(Debug, thiserror::Error)]
pub enum ImageError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Decode(#[from] png::DecodingError),
    #[error(transparent)]
    Encode(#[from] png::EncodingError),
    #[error("unsupported PNG color type {0:?}")]
    Unsupported(png::ColorType),
}

/// Load an 8-bit PNG (RGB, RGBA, gray, gray+alpha) as BGRA.
pub fn load_png(path: &Path) -> Result<BgraImage, ImageError> {
    let mut dec = png::Decoder::new(BufReader::new(File::open(path)?));
    dec.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = dec.read_info()?;
    let mut buf = vec![0; reader.output_buffer_size().unwrap_or(0)];
    let info = reader.next_frame(&mut buf)?;
    let (w, h) = (info.width, info.height);
    let mut img = BgraImage::new(w, h);
    let src = &buf[..info.buffer_size()];
    let out = img.data.as_chunks_mut::<4>().0.iter_mut();
    match info.color_type {
        png::ColorType::Rgba => out.zip(src.as_chunks::<4>().0).for_each(|(o, &[r, g, b, a])| *o = [b, g, r, a]),
        png::ColorType::Rgb => out.zip(src.as_chunks::<3>().0).for_each(|(o, &[r, g, b])| *o = [b, g, r, 255]),
        png::ColorType::Grayscale => out.zip(src).for_each(|(o, &g)| *o = [g, g, g, 255]),
        png::ColorType::GrayscaleAlpha => out.zip(src.as_chunks::<2>().0).for_each(|(o, &[g, a])| *o = [g, g, g, a]),
        other => return Err(ImageError::Unsupported(other)),
    }
    Ok(img)
}

/// Save BGRA as an opaque RGBA PNG.
pub fn save_png(path: &Path, img: BgraView<'_>) -> Result<(), ImageError> {
    let mut rgba = Vec::with_capacity(img.width as usize * img.height as usize * 4);
    for y in 0..img.height {
        for px in img.row(y).as_chunks::<4>().0 {
            rgba.extend_from_slice(&[px[2], px[1], px[0], 255]);
        }
    }
    let mut enc = png::Encoder::new(BufWriter::new(File::create(path)?), img.width, img.height);
    enc.set_color(png::ColorType::Rgba);
    enc.set_depth(png::BitDepth::Eight);
    enc.write_header()?.write_image_data(&rgba)?;
    Ok(())
}
