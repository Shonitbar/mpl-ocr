"""End-to-end check on a synthetic capture: 1-bit Arial 12 upscaled x1.5625 with bicubic,
which is the pipeline MapleStory Classic World's UI was measured to use.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from mapleocr_tools.gdi import Quality, render
from mapleocr_tools.identify_font import Candidate, analyze_roi, apply_key, estimate_scale, load_samples, ncc_max

SCALE = 1.5625
TEXT = "Bring back twenty mushroom caps quickly"


def synth_capture(tmp_path: Path) -> Path:
    glyphs = Image.fromarray(render("Arial", 12, TEXT, pad=1))
    up = glyphs.resize((round(glyphs.width * SCALE), round(glyphs.height * SCALE)), Image.Resampling.BICUBIC)
    a = np.asarray(up, dtype=np.float32) / 255.0
    bg = np.array([52, 60, 110], dtype=np.float32)  # tooltip-like dark blue
    frame = np.zeros((up.height + 20, up.width + 30, 3), dtype=np.float32) + bg
    frame[10 : 10 + up.height, 15 : 15 + up.width] = bg * (1 - a[..., None]) + 255 * a[..., None]
    Image.fromarray(frame.clip(0, 255).astype(np.uint8)).save(tmp_path / "frame.png")
    spec = {
        "schema": "mapleocr.font-samples/1",
        "samples": [
            {
                "roi": "tooltip",
                "image": "frame.png",
                "rect": [0, 0, frame.shape[1], frame.shape[0]],
                "text": TEXT,
            }
        ],
    }
    p = tmp_path / "samples.json"
    p.write_text(json.dumps(spec))
    return p


def test_ncc_finds_exact_template_and_handles_polarity() -> None:
    rng = np.random.default_rng(0)
    img = rng.uniform(0, 255, (40, 60)).astype(np.float32)
    tmpl = img[10:22, 20:45].copy()
    assert ncc_max(img, tmpl) == pytest.approx(1.0, abs=1e-4)
    assert ncc_max(img, 255 - tmpl) == pytest.approx(1.0, abs=1e-4)
    assert ncc_max(img[:5, :5], tmpl) == -1.0


def test_apply_key_color_distance() -> None:
    rgb = np.array([[[255, 255, 0], [0, 0, 0]]], dtype=np.uint8)
    k = apply_key(rgb, "#ffff00")
    assert k[0, 0] == 255 and k[0, 1] == 0


def test_identifies_font_scale_and_pipeline(tmp_path: Path) -> None:
    samples = load_samples(synth_capture(tmp_path))
    scale = estimate_scale(samples, range(10, 15), lambda _m: None)
    assert scale == pytest.approx(SCALE, abs=0.02)

    r = analyze_roi("tooltip", samples, ["Arial", "Tahoma", "Verdana", "Segoe UI"], range(10, 15), scale)
    assert r.best is not None
    assert (r.best["face"], r.best["px"], r.best["bold"]) == ("Arial", 12, False)
    assert r.verdict == "identified"
    assert max(r.source_scores, key=lambda k: r.source_scores[k]) == "nonantialiased"
    assert r.kernel_scores["bicubic"] > r.kernel_scores["nearest"]


def test_candidate_label() -> None:
    assert Candidate("Arial", 12, True, Quality.NONANTIALIASED).label() == "Arial 12px bold"
