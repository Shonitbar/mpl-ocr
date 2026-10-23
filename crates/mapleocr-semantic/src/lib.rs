//! Post-processing of recognized text: parsing into typed values and fuzzy
//! snapping against data packs.
//!
//! Packs are JSON files the user builds locally (e.g. with
//! `mapleocr pack add-map`); none ship with mpl-ocr, since map and item names
//! are game data.

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

/// Snap to a pack entry only if similarity is at least this.
pub const SNAP_THRESHOLD: f32 = 0.75;
/// Confidence reported for text that could not be checked against a pack.
pub const UNCHECKED_CONFIDENCE: f32 = 0.5;

#[derive(Debug, thiserror::Error)]
pub enum PackError {
    #[error(transparent)]
    Io(#[from] std::io::Error),
    #[error("{path}: {source}")]
    Json { path: PathBuf, source: serde_json::Error },
    #[error("{0}: unsupported pack schema {1:?}")]
    Schema(PathBuf, String),
}

#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct MapName {
    /// Street/region line, e.g. "Victoria Road". Empty if not shown.
    pub region: String,
    pub name: String,
}

pub const MAP_PACK_SCHEMA: &str = "mapleocr.pack.maps/1";

#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct MapPack {
    pub schema: String,
    pub maps: Vec<MapName>,
}

impl MapPack {
    pub fn load(path: &Path) -> Result<Self, PackError> {
        let text = std::fs::read_to_string(path)?;
        let pack: MapPack =
            serde_json::from_str(&text).map_err(|source| PackError::Json { path: path.to_owned(), source })?;
        if pack.schema != MAP_PACK_SCHEMA {
            return Err(PackError::Schema(path.to_owned(), pack.schema));
        }
        Ok(pack)
    }

    /// Load, or an empty pack if the file does not exist.
    pub fn load_or_default(path: &Path) -> Result<Self, PackError> {
        if path.exists() { Self::load(path) } else { Ok(Self { schema: MAP_PACK_SCHEMA.into(), maps: Vec::new() }) }
    }

    pub fn save(&self, path: &Path) -> Result<(), PackError> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let text =
            serde_json::to_string_pretty(self).map_err(|source| PackError::Json { path: path.to_owned(), source })?;
        std::fs::write(path, text)?;
        Ok(())
    }

    /// Add an entry unless an identical one exists. Returns true if added.
    pub fn add(&mut self, m: MapName) -> bool {
        if self.maps.contains(&m) {
            return false;
        }
        self.maps.push(m);
        true
    }

    /// Closest entry and its similarity in 0..=1.
    pub fn closest(&self, m: &MapName) -> Option<(&MapName, f32)> {
        let key = map_key(m);
        self.maps.iter().map(|e| (e, similarity(&key, &map_key(e)))).max_by(|a, b| a.1.total_cmp(&b.1))
    }
}

fn map_key(m: &MapName) -> String {
    format!("{}|{}", normalize(&m.region), normalize(&m.name))
}

/// Lowercase, collapse whitespace, and fold characters OCR commonly confuses
/// in this UI font (`l`/`I`/`|`, `0`/`O`).
pub fn normalize(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut space = false;
    for c in s.trim().chars() {
        if c.is_whitespace() {
            space = true;
            continue;
        }
        if space && !out.is_empty() {
            out.push(' ');
        }
        space = false;
        out.push(match c {
            'I' | '|' | '1' => 'l',
            '0' => 'o',
            c => c.to_ascii_lowercase(),
        });
    }
    out
}

/// Levenshtein distance over chars.
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let mut prev: Vec<usize> = (0..=b.len()).collect();
    let mut cur = vec![0; b.len() + 1];
    for (i, ca) in a.iter().enumerate() {
        cur[0] = i + 1;
        for (j, cb) in b.iter().enumerate() {
            cur[j + 1] = (prev[j + 1] + 1).min(cur[j] + 1).min(prev[j] + usize::from(ca != cb));
        }
        std::mem::swap(&mut prev, &mut cur);
    }
    prev[b.len()]
}

/// 1 - normalized edit distance.
pub fn similarity(a: &str, b: &str) -> f32 {
    let n = a.chars().count().max(b.chars().count());
    if n == 0 { 1.0 } else { 1.0 - levenshtein(a, b) as f32 / n as f32 }
}

#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Parsed<T> {
    pub value: T,
    pub confidence: f32,
    /// True if the value was replaced by a pack entry.
    pub snapped: bool,
}

/// Parse the minimap's text lines ("region" then "map name") and snap them
/// to the pack if one is given and close enough.
pub fn parse_map_name(lines: &[String], pack: Option<&MapPack>) -> Option<Parsed<MapName>> {
    let lines: Vec<&str> = lines.iter().map(|l| l.trim()).filter(|l| !l.is_empty()).collect();
    let raw = match lines.as_slice() {
        [] => return None,
        [name] => MapName { region: String::new(), name: (*name).to_owned() },
        [region, rest @ ..] => MapName { region: (*region).to_owned(), name: rest.join(" ") },
    };
    if let Some((entry, sim)) = pack.and_then(|p| p.closest(&raw))
        && sim >= SNAP_THRESHOLD
    {
        return Some(Parsed { value: entry.clone(), confidence: sim, snapped: true });
    }
    Some(Parsed { value: raw, confidence: UNCHECKED_CONFIDENCE, snapped: false })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn m(r: &str, n: &str) -> MapName {
        MapName { region: r.into(), name: n.into() }
    }

    #[test]
    fn edit_distance() {
        assert_eq!(levenshtein("kitten", "sitting"), 3);
        assert_eq!(levenshtein("", "abc"), 3);
        assert!((similarity("henesys", "henesys") - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn normalize_folds_ocr_confusions() {
        assert_eq!(normalize("  Hidden   Street "), "hidden street");
        assert_eq!(normalize("ElIinia"), normalize("Ellinia"));
        assert_eq!(normalize("Ludibrium 0"), "ludibrium o");
    }

    #[test]
    fn parses_and_snaps() {
        let pack = MapPack {
            schema: MAP_PACK_SCHEMA.into(),
            maps: vec![
                m("Victoria Road", "Henesys Market"),
                m("Victoria Road", "Henesys"),
                m("Hidden Street", "Free Market Entrance"),
            ],
        };
        let lines = vec!["Victoria Raad".to_string(), "Henesys Markot".to_string()];
        let p = parse_map_name(&lines, Some(&pack)).unwrap();
        assert!(p.snapped);
        assert_eq!(p.value, m("Victoria Road", "Henesys Market"));

        let p = parse_map_name(&["Victoria Road".into(), "Henesys".into()], Some(&pack)).unwrap();
        assert_eq!(p.value, m("Victoria Road", "Henesys"));
        assert!((p.confidence - 1.0).abs() < 1e-6);

        let p = parse_map_name(&["Ludibrium".into(), "Toy Factory".into()], Some(&pack)).unwrap();
        assert!(!p.snapped);
        assert_eq!(p.value, m("Ludibrium", "Toy Factory"));
        assert!(parse_map_name(&[], None).is_none());
    }
}
