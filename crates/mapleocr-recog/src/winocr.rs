//! Fallback OCR tier: the `Windows.Media.Ocr` engine built into Windows 10/11.
//!
//! No model download, no extra runtime, ~10-40 ms per line. It is the
//! general-purpose fallback until the ONNX tiers land; it never runs on the
//! capture thread.

use mapleocr_imgproc::{BgraView, Rect};
use windows::Globalization::Language;
use windows::Graphics::Imaging::{BitmapPixelFormat, SoftwareBitmap};
use windows::Media::Ocr::OcrEngine;
use windows::Security::Cryptography::CryptographicBuffer;
use windows::Win32::System::WinRT::{RO_INIT_MULTITHREADED, RoInitialize};
use windows::core::HSTRING;

use crate::{RecogError, Recognition, Recognizer, Result};

pub struct WinOcr {
    engine: OcrEngine,
    /// Scratch buffer for tightly packed pixels.
    packed: Vec<u8>,
}

impl WinOcr {
    /// Create an English OCR engine. Requires the English language pack
    /// (present on English Windows installs).
    pub fn new() -> Result<Self> {
        // SAFETY: idempotent per thread; an existing apartment is fine too.
        let _ = unsafe { RoInitialize(RO_INIT_MULTITHREADED) };
        let lang = Language::CreateLanguage(&HSTRING::from("en-US"))?;
        if !OcrEngine::IsLanguageSupported(&lang)? {
            return Err(RecogError::Unavailable(
                "Windows OCR has no English support; add the English language pack in Settings".into(),
            ));
        }
        let engine = OcrEngine::TryCreateFromLanguage(&lang)?;
        Ok(Self { engine, packed: Vec::new() })
    }

    pub fn max_image_dimension() -> u32 {
        OcrEngine::MaxImageDimension().unwrap_or(10_000)
    }
}

impl Recognizer for WinOcr {
    fn engine(&self) -> &'static str {
        "winocr"
    }

    fn recognize(&mut self, img: BgraView<'_>) -> Result<Recognition> {
        if img.width == 0 || img.height == 0 {
            return Ok(Recognition::default());
        }
        self.packed.clear();
        for y in 0..img.height {
            self.packed.extend_from_slice(img.row(y));
        }
        let buffer = CryptographicBuffer::CreateFromByteArray(&self.packed)?;
        let bitmap = SoftwareBitmap::CreateCopyFromBuffer(
            &buffer,
            BitmapPixelFormat::Bgra8,
            img.width as i32,
            img.height as i32,
        )?;
        let result = self.engine.RecognizeAsync(&bitmap)?.join()?;
        let (mut lines, mut boxes) = (Vec::new(), Vec::new());
        for line in result.Lines()? {
            lines.push(line.Text()?.to_string());
            let (mut x0, mut y0, mut x1, mut y1) = (f32::MAX, f32::MAX, f32::MIN, f32::MIN);
            for w in line.Words()? {
                let r = w.BoundingRect()?;
                (x0, y0, x1, y1) = (x0.min(r.X), y0.min(r.Y), x1.max(r.X + r.Width), y1.max(r.Y + r.Height));
            }
            boxes.push(if x0 <= x1 {
                Rect::new(x0 as i32, y0 as i32, (x1 - x0).ceil() as u32, (y1 - y0).ceil() as u32)
            } else {
                Rect::default()
            });
        }
        Ok(Recognition { lines, boxes, confidence: None })
    }
}
