//! Declarative ROI registry and anchor-based auto-calibration.
//!
//! A [`Layout`] describes UI landmarks (anchors) and regions of interest in
//! *native UI units*. [`calibrate`] turns it into client-pixel rectangles for
//! one window size, locating each anchor by template match when the user has
//! anchor templates (cut from their own screen by [`AnchorTemplates::bootstrap`]),
//! and falling back to the declared default position otherwise.

use std::collections::BTreeMap;
use std::path::Path;

use mapleocr_imgproc::{BgraView, Gray, Rect, ncc_search};
use serde::{Deserialize, Serialize};

/// Minimum NCC to accept a template match over the default position.
pub const MATCH_THRESHOLD: f32 = 0.75;

#[derive(Debug, thiserror::Error)]
pub enum LayoutError {
    #[error("layout YAML: {0}")]
    Yaml(#[from] serde_yaml_ng::Error),
    #[error("layout: {0}")]
    Invalid(String),
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error(transparent)]
    Image(#[from] mapleocr_imgproc::ImageError),
    #[error(transparent)]
    Json(#[from] serde_json::Error),
}

pub type Result<T, E = LayoutError> = std::result::Result<T, E>;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Origin {
    #[default]
    TopLeft,
    TopCenter,
    TopRight,
    BottomLeft,
    BottomCenter,
    BottomRight,
}

impl Origin {
    fn point(self, (w, h): (u32, u32)) -> (f32, f32) {
        let (w, h) = (w as f32, h as f32);
        match self {
            Self::TopLeft => (0.0, 0.0),
            Self::TopCenter => (w / 2.0, 0.0),
            Self::TopRight => (w, 0.0),
            Self::BottomLeft => (0.0, h),
            Self::BottomCenter => (w / 2.0, h),
            Self::BottomRight => (w, h),
        }
    }
}

/// `[x, y, w, h]` in native UI units.
pub type NativeRect = [f32; 4];

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct AnchorSpec {
    #[serde(default)]
    pub origin: Origin,
    /// Default position, relative to `origin`.
    pub rect: NativeRect,
    /// Where the anchor's top-left may be, relative to `origin`. Defaults to
    /// the default rect grown by 16 native units on each side.
    #[serde(default)]
    pub search: Option<NativeRect>,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RoiSpec {
    pub anchor: String,
    /// Relative to the anchor's top-left.
    pub rect: NativeRect,
    /// Recognizer tier: `ocr` or `glyph`.
    pub recognizer: String,
    /// Glyph font: `gdi:<face>:<px>[:bold]` or `learned:<name>`.
    #[serde(default)]
    pub font: Option<String>,
    /// Color key applied before recognition: `luma`, `white`, `#rrggbb`.
    #[serde(default)]
    pub key: Option<String>,
    /// Content-aware crop applied before recognition, e.g. `panel`.
    #[serde(default)]
    pub crop: Option<String>,
    /// Semantic parser for the recognized text, e.g. `map_name`.
    #[serde(default)]
    pub parse: Option<String>,
    /// Recognize only after the ROI's pixels are unchanged for this many frames.
    #[serde(default)]
    pub settle_frames: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Layout {
    pub schema: String,
    pub canvas_height: f32,
    pub anchors: BTreeMap<String, AnchorSpec>,
    pub rois: BTreeMap<String, RoiSpec>,
}

impl Layout {
    pub fn from_yaml(s: &str) -> Result<Self> {
        let l: Layout = serde_yaml_ng::from_str(s)?;
        if l.schema != "mapleocr.layout/1" {
            return Err(LayoutError::Invalid(format!("unsupported schema {:?}", l.schema)));
        }
        if l.canvas_height <= 0.0 {
            return Err(LayoutError::Invalid("canvas_height must be positive".into()));
        }
        for (name, roi) in &l.rois {
            if !l.anchors.contains_key(&roi.anchor) {
                return Err(LayoutError::Invalid(format!("ROI {name} uses unknown anchor {:?}", roi.anchor)));
            }
        }
        Ok(l)
    }

    pub fn load(path: &Path) -> Result<Self> {
        Self::from_yaml(&std::fs::read_to_string(path)?)
    }

    /// The built-in MapleStory Classic World layout.
    pub fn builtin() -> Self {
        Self::from_yaml(include_str!("default_layout.yaml")).expect("built-in layout is valid")
    }

    pub fn scale_for(&self, client: (u32, u32)) -> f32 {
        client.1 as f32 / self.canvas_height
    }

    fn to_client(&self, origin: Origin, r: NativeRect, client: (u32, u32)) -> Rect {
        let s = self.scale_for(client);
        let (ox, oy) = origin.point(client);
        Rect::new(
            (ox + r[0] * s).round() as i32,
            (oy + r[1] * s).round() as i32,
            (r[2] * s).round().max(1.0) as u32,
            (r[3] * s).round().max(1.0) as u32,
        )
    }

    pub fn default_anchor_rect(&self, name: &str, client: (u32, u32)) -> Option<Rect> {
        let a = self.anchors.get(name)?;
        Some(self.to_client(a.origin, a.rect, client))
    }
}

/// Anchor templates cut from the user's own screen, plus the scale they were
/// captured at. Stored as PNGs so users can inspect them.
#[derive(Clone, Debug, Default)]
pub struct AnchorTemplates {
    pub scale: f32,
    pub images: BTreeMap<String, Gray>,
}

#[derive(Serialize, Deserialize)]
struct TemplatesMeta {
    schema: String,
    scale: f32,
    anchors: Vec<String>,
}

impl AnchorTemplates {
    /// Cut every anchor at its default position. Call when the UI is in its
    /// default arrangement (e.g. right after login).
    pub fn bootstrap(layout: &Layout, frame: BgraView<'_>) -> Self {
        let client = (frame.width, frame.height);
        let images = layout
            .anchors
            .keys()
            .filter_map(|name| {
                let r = layout.default_anchor_rect(name, client)?;
                Some((name.clone(), Gray::from_bgra(frame.crop(r))))
            })
            .collect();
        Self { scale: layout.scale_for(client), images }
    }

    pub fn save(&self, dir: &Path) -> Result<()> {
        std::fs::create_dir_all(dir)?;
        for (name, g) in &self.images {
            let mut bgra = mapleocr_imgproc::BgraImage::new(g.width, g.height);
            for (px, &v) in bgra.data.as_chunks_mut::<4>().0.iter_mut().zip(&g.data) {
                let v = v.round().clamp(0.0, 255.0) as u8;
                *px = [v, v, v, 255];
            }
            mapleocr_imgproc::save_png(&dir.join(format!("{name}.png")), bgra.view())?;
        }
        let meta = TemplatesMeta {
            schema: "mapleocr.anchors/1".into(),
            scale: self.scale,
            anchors: self.images.keys().cloned().collect(),
        };
        std::fs::write(dir.join("anchors.json"), serde_json::to_string_pretty(&meta)?)?;
        Ok(())
    }

    /// `Ok(None)` if no templates were saved yet.
    pub fn load(dir: &Path) -> Result<Option<Self>> {
        let meta_path = dir.join("anchors.json");
        if !meta_path.exists() {
            return Ok(None);
        }
        let meta: TemplatesMeta = serde_json::from_str(&std::fs::read_to_string(meta_path)?)?;
        let mut images = BTreeMap::new();
        for name in meta.anchors {
            let img = mapleocr_imgproc::load_png(&dir.join(format!("{name}.png")))?;
            images.insert(name, Gray::from_bgra(img.view()));
        }
        Ok(Some(Self { scale: meta.scale, images }))
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub enum FixSource {
    /// Template matched above [`MATCH_THRESHOLD`].
    Matched,
    /// No template, or the match was too weak: declared default position.
    Default,
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize)]
pub struct AnchorFix {
    pub rect: Rect,
    pub source: FixSource,
    /// Best NCC found, if a template was searched.
    pub score: Option<f32>,
}

/// A layout resolved for one client size.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Calibration {
    pub client_size: (u32, u32),
    pub scale: f32,
    pub anchors: BTreeMap<String, AnchorFix>,
    /// ROI rectangles in client pixels, clamped to the client area.
    pub rois: BTreeMap<String, Rect>,
}

impl Calibration {
    /// True if every anchor that has a template was found by matching.
    pub fn all_matched(&self) -> bool {
        self.anchors.values().all(|a| a.score.is_none() || a.source == FixSource::Matched)
    }
}

/// Resolve `layout` for the client size of `frame` (a full client-area frame).
pub fn calibrate(layout: &Layout, frame: BgraView<'_>, templates: Option<&AnchorTemplates>) -> Calibration {
    let client = (frame.width, frame.height);
    let scale = layout.scale_for(client);
    let gray = templates.map(|_| Gray::from_bgra(frame));
    let mut anchors = BTreeMap::new();
    for (name, spec) in &layout.anchors {
        let default = layout.to_client(spec.origin, spec.rect, client);
        let mut fix = AnchorFix { rect: default, source: FixSource::Default, score: None };
        if let (Some(t), Some(gray)) = (templates.and_then(|t| t.images.get(name).map(|g| (t, g))), &gray) {
            let (ts, tmpl) = t;
            let k = scale / ts.scale;
            let tmpl = if (k - 1.0).abs() < 1e-3 {
                tmpl.clone()
            } else {
                tmpl.resize((tmpl.width as f32 * k).round() as u32, (tmpl.height as f32 * k).round() as u32)
            };
            let search = spec.search.unwrap_or([
                spec.rect[0] - 16.0,
                spec.rect[1] - 16.0,
                spec.rect[2] + 32.0,
                spec.rect[3] + 32.0,
            ]);
            let area = layout.to_client(spec.origin, search, client);
            if let Some(m) = ncc_search(gray, &tmpl, area) {
                fix.score = Some(m.score);
                if m.score >= MATCH_THRESHOLD {
                    fix = AnchorFix {
                        rect: Rect::new(m.x, m.y, tmpl.width, tmpl.height),
                        source: FixSource::Matched,
                        score: Some(m.score),
                    };
                }
            }
        }
        anchors.insert(name.clone(), fix);
    }
    let rois = layout
        .rois
        .iter()
        .map(|(name, roi)| {
            let a = anchors[&roi.anchor].rect;
            let r = Rect::new(
                a.x + (roi.rect[0] * scale).round() as i32,
                a.y + (roi.rect[1] * scale).round() as i32,
                (roi.rect[2] * scale).round().max(1.0) as u32,
                (roi.rect[3] * scale).round().max(1.0) as u32,
            );
            (name.clone(), r.clamp_to(client.0, client.1))
        })
        .collect();
    Calibration { client_size: client, scale, anchors, rois }
}

#[cfg(test)]
mod tests {
    use super::*;
    use mapleocr_imgproc::BgraImage;

    /// Synthetic "UI": a textured caption box drawn at a given offset.
    fn frame(w: u32, h: u32, at: (u32, u32), box_size: (u32, u32)) -> BgraImage {
        let mut img = BgraImage::new(w, h);
        for y in 0..h {
            for x in 0..w {
                let inside = x >= at.0 && x < at.0 + box_size.0 && y >= at.1 && y < at.1 + box_size.1;
                let v = if inside { (((x - at.0) * 7 + (y - at.1) * 13) % 200) as u8 + 40 } else { 90 };
                let o = ((y * w + x) * 4) as usize;
                img.data[o..o + 4].copy_from_slice(&[v, v, v, 255]);
            }
        }
        img
    }

    #[test]
    fn builtin_layout_parses_and_scales() {
        let l = Layout::builtin();
        assert_eq!(l.scale_for((1920, 1200)), 1.5625);
        let r = l.default_anchor_rect("minimap_caption", (1920, 1200)).unwrap();
        assert_eq!((r.x, r.y), (9, 11));
    }

    #[test]
    fn rejects_unknown_anchor() {
        let y = "schema: mapleocr.layout/1\ncanvas_height: 768\nanchors: {}\nrois:\n  a: {anchor: nope, rect: [0,0,1,1], recognizer: ocr}\n";
        assert!(Layout::from_yaml(y).is_err());
    }

    #[test]
    fn calibration_follows_a_moved_anchor() {
        let layout = Layout::from_yaml(
            "schema: mapleocr.layout/1\ncanvas_height: 100\nanchors:\n  cap: {rect: [10, 10, 40, 10], search: [0, 0, 100, 60]}\nrois:\n  r: {anchor: cap, rect: [0, 12, 30, 5], recognizer: ocr}\n",
        )
        .unwrap();
        // Bootstrap at scale 2 with the caption at its default spot (20, 20).
        let boot = frame(400, 200, (20, 20), (80, 20));
        let t = AnchorTemplates::bootstrap(&layout, boot.view());
        assert_eq!(t.scale, 2.0);
        // Same scale, caption moved to (60, 50).
        let moved = frame(400, 200, (60, 50), (80, 20));
        let c = calibrate(&layout, moved.view(), Some(&t));
        let a = c.anchors["cap"];
        assert_eq!(a.source, FixSource::Matched);
        assert_eq!((a.rect.x, a.rect.y), (60, 50));
        assert_eq!(c.rois["r"], Rect::new(60, 74, 60, 10));
        // Without templates: default position.
        let c = calibrate(&layout, moved.view(), None);
        assert_eq!(c.anchors["cap"].source, FixSource::Default);
        assert_eq!(c.rois["r"], Rect::new(20, 44, 60, 10));
    }
}
