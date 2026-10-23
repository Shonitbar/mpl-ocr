"""identify-font: work out which font and rendering pipeline produced on-screen text.

Given labeled text crops from `mapleocr capture-dataset` frames, this:

1. Estimates the UI scale factor: the ratio between the size the client
   draws text at and the pixels on screen.
2. Ranks candidate fonts by rendering each word with GDI (1-bit, the way a
   classic Win32 client draws text), upscaling it by that factor, and taking
   the best normalized cross-correlation against the capture.
3. Diagnoses the pipeline for the best match: was the source text 1-bit or
   anti-aliased, and was it upscaled with nearest-neighbour or a smooth filter?
4. Suggests a recognizer per ROI.

Samples file (JSON, paths relative to it):

    {
      "schema": "mapleocr.font-samples/1",
      "samples": [
        {"roi": "tooltip", "image": "session-1/0006_full.png",
         "rect": [1136, 362, 300, 42], "text": "the exact text in the rect",
         "key": "luma"}
      ]
    }

`key` turns the crop into one intensity channel: `luma` (default), `white`
(bright, unsaturated pixels only), or `#rrggbb` (distance to a text color;
use this for text over translucent or busy backgrounds).

Usage:
    python -m mapleocr_tools.identify_font samples.json [--all-fonts] [--out report.json]
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageFont

from mapleocr_tools.gdi import Quality, font_families, load_private_font, render

F32 = npt.NDArray[np.float32]

# Fonts a Windows game UI of the 2000s-2020s plausibly uses. `--all-fonts` scans every family.
COMMON_FACES = [
    "Arial",
    "Arial Narrow",
    "Tahoma",
    "Verdana",
    "Microsoft Sans Serif",
    "MS Sans Serif",
    "Segoe UI",
    "Calibri",
    "Trebuchet MS",
    "Lucida Sans Unicode",
    "Century Gothic",
    "Gulim",
    "Dotum",
    "Malgun Gothic",
    "MS Gothic",
    "SimSun",
    "Courier New",
    "Consolas",
]
PROBE_FACES = ["Arial", "Tahoma", "Verdana", "Microsoft Sans Serif", "Segoe UI", "Calibri"]
KERNELS = {
    "nearest": Image.Resampling.NEAREST,
    "bilinear": Image.Resampling.BILINEAR,
    "bicubic": Image.Resampling.BICUBIC,
    "lanczos": Image.Resampling.LANCZOS,
}


@dataclass(frozen=True)
class Candidate:
    face: str
    px: int
    bold: bool
    quality: Quality = Quality.NONANTIALIASED

    def label(self) -> str:
        return f"{self.face} {self.px}px {'bold' if self.bold else 'regular'}"


@dataclass
class Sample:
    roi: str
    text: str
    image: Path
    rect: tuple[int, int, int, int]
    key: str
    pixels: F32 = field(repr=False)

    def words(self) -> list[str]:
        """Words long enough to be distinctive; short lines use the whole text."""
        ws = [w for w in self.text.split() if len(w) >= 3]
        return ws or [self.text]


# ---------------------------------------------------------------- image ops


def apply_key(rgb: npt.NDArray[np.uint8], key: str) -> F32:
    a = rgb.astype(np.float32)
    if key == "luma":
        return a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
    if key == "white":
        lo, hi = a.min(axis=2), a.max(axis=2)
        return np.clip((lo - 150.0) * 2.5, 0, 255) * (hi - lo < 60)
    if key.startswith("#") and len(key) == 7:
        color = np.array([int(key[i : i + 2], 16) for i in (1, 3, 5)], dtype=np.float32)
        dist = np.sqrt(((a - color) ** 2).sum(axis=2))
        return np.clip(255.0 - dist * 2.0, 0, 255).astype(np.float32)
    raise ValueError(f"unknown key {key!r}; use luma, white or #rrggbb")


def ncc_max(image: F32, tmpl: F32) -> float:
    """Best |normalized cross-correlation| of `tmpl` over every position in `image`.

    Absolute value so dark-on-light and light-on-dark text both match.
    FFT for the numerator, integral images for the window statistics.
    """
    th, tw = tmpl.shape
    ih, iw = image.shape
    if th > ih or tw > iw:
        return -1.0
    t = tmpl - tmpl.mean()
    tn = float(np.sqrt((t * t).sum()))
    if tn == 0.0:
        return -1.0
    # Circular convolution of size (ih, iw) is alias-free for the valid region.
    num = np.fft.irfft2(np.fft.rfft2(image) * np.fft.rfft2(t[::-1, ::-1], s=image.shape), s=image.shape)
    num = num[th - 1 :, tw - 1 :]

    def window_sums(x: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        s = np.pad(x.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
        return s[th:, tw:] - s[:-th, tw:] - s[th:, :-tw] + s[:-th, :-tw]

    img64 = image.astype(np.float64)
    n = th * tw
    s1 = window_sums(img64)
    var = window_sums(img64 * img64) - s1 * s1 / n
    den = np.sqrt(np.maximum(var, 0.0)) * tn
    valid = den > 1e-6 * tn * 255
    if not valid.any():
        return -1.0
    return float(np.abs(num[valid] / den[valid]).max())


@lru_cache(maxsize=200_000)
def _glyphs(c: Candidate, word: str) -> npt.NDArray[np.uint8]:
    return render(c.face, c.px, word, bold=c.bold, quality=c.quality, pad=1)


def template(c: Candidate, word: str, scale: float, kernel: str = "bicubic") -> F32:
    src = Image.fromarray(_glyphs(c, word))
    size = (max(1, round(src.width * scale)), max(1, round(src.height * scale)))
    return np.asarray(src.resize(size, KERNELS[kernel]), dtype=np.float32)


def score(s: Sample, c: Candidate, scale: float, kernel: str = "bicubic") -> float:
    return float(np.mean([ncc_max(s.pixels, template(c, w, scale, kernel)) for w in s.words()]))


# ---------------------------------------------------------------- pipeline


def load_samples(path: Path) -> list[Sample]:
    spec = json.loads(path.read_text(encoding="utf-8"))
    if spec.get("schema") != "mapleocr.font-samples/1":
        raise ValueError(f"{path}: expected schema mapleocr.font-samples/1")
    out: list[Sample] = []
    for i, e in enumerate(spec["samples"]):
        img_path = (path.parent / e["image"]).resolve()
        x, y, w, h = (int(v) for v in e["rect"])
        rgb = np.asarray(Image.open(img_path).convert("RGB"))[y : y + h, x : x + w]
        if rgb.shape[0] != h or rgb.shape[1] != w:
            raise ValueError(f"sample {i}: rect {e['rect']} falls outside {img_path.name}")
        key = e.get("key", "luma")
        out.append(Sample(e["roi"], e["text"], img_path, (x, y, w, h), key, apply_key(rgb, key)))
    return out


def estimate_scale(samples: list[Sample], sizes: range, log: Any) -> float:
    """Coarse scale scan with common faces, then a fine scan around the best."""
    probe = samples[:2]
    cands = [Candidate(f, px, b) for f in PROBE_FACES for px in sizes for b in (False, True)]
    coarse: list[tuple[float, float, Candidate]] = []
    for scale in np.arange(1.0, 2.0001, 0.0625):
        for c in cands:
            coarse.append((float(np.mean([score(s, c, float(scale)) for s in probe])), float(scale), c))
    coarse.sort(key=lambda t: -t[0])
    center = coarse[0][1]
    log(f"  coarse: best {coarse[0][2].label()} at x{center:.4f} (ncc {coarse[0][0]:.3f})")
    # Different ROIs use different fonts: refine with each sample's own best candidate.
    own = [max(cands, key=lambda c, s=s: score(s, c, center)) for s in samples]
    fine: list[tuple[float, float]] = []
    for scale in np.arange(center - 0.0625, center + 0.0626, 0.0025):
        fine.append(
            (float(np.mean([score(s, c, float(scale)) for s, c in zip(samples, own, strict=True)])), float(scale))
        )
    best_ncc, best_scale = max(fine)
    log(f"  fine:   x{best_scale:.4f} (mean ncc {best_ncc:.3f} over {len(samples)} samples)")
    return best_scale


@dataclass
class RoiResult:
    roi: str
    samples: int
    ranking: list[dict[str, Any]]
    best: dict[str, Any] | None
    verdict: str
    margin: float
    kernel_scores: dict[str, float]
    source_scores: dict[str, float]
    suggested_engine: str


def analyze_roi(roi: str, samples: list[Sample], faces: list[str], sizes: range, scale: float) -> RoiResult:
    cands = [Candidate(f, px, b) for f in faces for px in sizes for b in (False, True)]
    scored = sorted(((float(np.mean([score(s, c, scale) for s in samples])), c) for c in cands), key=lambda t: -t[0])
    ranking = [{"ncc": round(n, 4), **asdict(c), "quality": c.quality.name.lower()} for n, c in scored[:10]]
    best_ncc, best = scored[0]
    other_family = next((n for n, c in scored if c.face != best.face), 0.0)
    margin = best_ncc - other_family

    def mean_score(c: Candidate, kernel: str = "bicubic") -> float:
        return round(float(np.mean([score(s, c, scale, kernel) for s in samples])), 4)

    kernel_scores = {k: mean_score(best, k) for k in KERNELS}
    source_scores = {
        q.name.lower(): mean_score(Candidate(best.face, best.px, best.bold, q))
        for q in (Quality.NONANTIALIASED, Quality.ANTIALIASED, Quality.CLEARTYPE)
    }
    if best_ncc >= 0.9 and margin >= 0.05:
        verdict = "identified"
    elif best_ncc >= 0.8:
        verdict = "probable"
    else:
        verdict = "unresolved"
    one_bit = source_scores["nonantialiased"] >= max(source_scores.values()) - 0.005
    if verdict == "unresolved":
        engine = "crnn, or glyph atlas built from captures (custom/sprite font or font not installed)"
    elif one_bit:
        engine = "glyph (1-bit source font: resample ROI to the native grid, then exact template match)"
    else:
        engine = "glyph with grayscale templates, or crnn"
    return RoiResult(
        roi,
        len(samples),
        ranking,
        ranking[0],
        verdict,
        round(margin, 4),
        kernel_scores,
        source_scores,
        engine,
    )


def client_dir_fonts(client_dir: Path, log: Any) -> list[str]:
    """Load font files shipped with the user's own client, privately, and return their families."""
    faces: list[str] = []
    for p in sorted(client_dir.rglob("*")):
        if p.suffix.lower() not in {".ttf", ".ttc", ".otf", ".fon", ".fnt"}:
            continue
        log(f"  client font file: {p}")
        if not load_private_font(p):
            continue
        # Bitmap .fon/.fnt files render via GDI but FreeType can't name them.
        with contextlib.suppress(OSError):
            name = ImageFont.truetype(str(p), 12).getname()[0]
            if name:
                faces.append(name)
    return faces


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="identify-font", description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("samples", type=Path, help="samples JSON (mapleocr.font-samples/1)")
    ap.add_argument("--scale", type=float, help="known UI scale; skips estimation")
    ap.add_argument("--sizes", default="9-16", help="em sizes in px to try, e.g. 9-16")
    ap.add_argument("--font", action="append", default=[], help="extra face to try (repeatable)")
    ap.add_argument("--all-fonts", action="store_true", help="try every installed family (slow)")
    ap.add_argument("--client-dir", type=Path, help="scan your own client install for font files")
    ap.add_argument("--out", type=Path, help="write the JSON report here")
    args = ap.parse_args(argv)

    def log(msg: str) -> None:
        print(msg, file=sys.stderr, flush=True)

    lo, hi = (int(v) for v in args.sizes.split("-"))
    sizes = range(lo, hi + 1)
    t0 = time.perf_counter()
    samples = load_samples(args.samples)
    log(f"{len(samples)} samples, {len({s.roi for s in samples})} ROI types")

    installed = set(font_families())
    faces = list(dict.fromkeys([*COMMON_FACES, *args.font]))
    if args.client_dir:
        faces += client_dir_fonts(args.client_dir, log)
        installed = set(font_families())
    if args.all_fonts:
        faces = sorted(installed)
    missing = [f for f in faces if f not in installed]
    faces = [f for f in faces if f in installed]
    if missing:
        log(f"not installed, skipped: {', '.join(missing)}")

    if args.scale:
        scale = args.scale
        log(f"scale: x{scale:.4f} (given)")
    else:
        log("estimating UI scale...")
        scale = estimate_scale(samples, sizes, log)

    by_roi: dict[str, list[Sample]] = {}
    for s in samples:
        by_roi.setdefault(s.roi, []).append(s)
    results: list[RoiResult] = []
    for roi, ss in by_roi.items():
        log(f"ranking {len(faces)} faces x {len(sizes)} sizes x 2 weights for {roi!r}...")
        results.append(analyze_roi(roi, ss, faces, sizes, scale))

    print(f"\nUI scale: x{scale:.4f}  (native text height = screen height / {scale:.4f})")
    print(f"{'roi':<16} {'verdict':<11} {'best match':<34} {'ncc':>6} {'margin':>7}  source / upscale")
    for r in results:
        assert r.best is not None
        b = r.best
        label = f"{b['face']} {b['px']}px {'bold' if b['bold'] else 'regular'}"
        src = max(r.source_scores, key=lambda k: r.source_scores[k])
        ker = max(r.kernel_scores, key=lambda k: r.kernel_scores[k])
        print(f"{r.roi:<16} {r.verdict:<11} {label:<34} {b['ncc']:>6.3f} {r.margin:>7.3f}  {src} / {ker}")
    print()
    for r in results:
        print(f"{r.roi}: {r.suggested_engine}")

    report = {
        "schema": "mapleocr.font-report/1",
        "scale": round(scale, 5),
        "faces_tried": len(faces),
        "sizes": [lo, hi],
        "seconds": round(time.perf_counter() - t0, 1),
        "rois": [asdict(r) for r in results],
    }
    if args.out:
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        log(f"report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
