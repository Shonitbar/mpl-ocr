"""mpl-ocr vs. standard Python OCR on the same saved game frames.

    bench/.venv/Scripts/python bench/compare.py [--frames N] [--engines rapid,tesseract,easy]

Needs `mapleocr` on PATH, `fixtures/local/labels.json` (your own labeled
captures, see docs/design.md) and `pip install -r bench/requirements.txt`.

What is measured, per engine:
  * ROI crop: the exact HP/MP/EXP/map-name rectangle that mpl-ocr calibrated is
    handed to the Python engine. This is the BEST case for Python: a real app would
    also have to find that rectangle first.
  * Full frame: the whole 1920x1200 client frame; the answer counts as correct if the
    expected text appears anywhere in the output (lenient).
  * Item tooltips: full frame; correct if the item name appears in the output.
Times are wall-clock per call after warm-up, on this machine.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
LABELS = ROOT / "fixtures" / "local" / "labels.json"
TESSERACT = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Item names visible in the tooltip frames of the labeled session.
TOOLTIP_TRUTH = {
    "0004_full.png": ["Brown Skullcap", "Black Bandana"],
    "0005_full.png": ["Blue Pao", "Grey T-Shirt"],
    "0006_full.png": ["Leaf"],
}


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9/%.]", "", s.lower())


def digits_fold(s: str) -> str:
    return s.translate(str.maketrans("OolI|SBZ", "00111582"))


def pct(v: list[float], q: float) -> float:
    v = sorted(v)
    return v[min(len(v) - 1, round((len(v) - 1) * q))]


class Engine:
    name = "?"

    def read(self, rgb: np.ndarray) -> str:
        raise NotImplementedError


class Rapid(Engine):
    name = "RapidOCR (PaddleOCR on ONNX)"

    def __init__(self) -> None:
        from rapidocr_onnxruntime import RapidOCR

        self.e = RapidOCR()

    def read(self, rgb: np.ndarray) -> str:
        res, _ = self.e(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        return " ".join(r[1] for r in res) if res else ""


class Tesseract(Engine):
    name = "Tesseract (pytesseract)"

    def __init__(self) -> None:
        import pytesseract

        pytesseract.pytesseract.tesseract_cmd = TESSERACT
        self.t = pytesseract

    def read(self, rgb: np.ndarray) -> str:
        g = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        if g.shape[0] < 200:  # small ROI: the standard recipe is upscale, invert if light-on-dark, Otsu
            g = cv2.resize(g, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            if g.mean() < 128:
                g = 255 - g
            g = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
            return self.t.image_to_string(g, config="--psm 7")
        return self.t.image_to_string(g, config="--psm 11")


class Easy(Engine):
    name = "EasyOCR (CPU)"

    def __init__(self) -> None:
        import easyocr

        self.e = easyocr.Reader(["en"], gpu=False, verbose=False)

    def read(self, rgb: np.ndarray) -> str:
        return " ".join(self.e.readtext(rgb, detail=0))


def mapleocr_json(frame: Path, rois: list[str]) -> dict:
    args = ["mapleocr", "recognize", str(frame), "--json"]
    for r in rois:
        args += ["--roi", r]
    out = subprocess.run(args, capture_output=True, text=True, check=True).stdout
    return json.loads(out.strip().splitlines()[0])


def hp_ok(text: str, truth: str) -> bool:
    cur, mx = truth.split("/")
    m = re.findall(r"(\d+)\s*[/7|]\s*(\d+)", digits_fold(text))
    return any(a == cur and b == mx for a, b in m)


def timed(fn, *a) -> tuple[float, str]:
    t = time.perf_counter()
    r = fn(*a)
    return (time.perf_counter() - t) * 1000, r


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=12, help="ROI frames per engine")
    ap.add_argument("--full-frames", type=int, default=5, help="full-frame calls per engine")
    ap.add_argument("--engines", default="rapid,tesseract,easy")
    args = ap.parse_args()
    if not shutil.which("mapleocr"):
        sys.exit("mapleocr is not on PATH (cargo install --path crates/mapleocr-cli --locked)")

    labels = json.loads(LABELS.read_text())
    base = LABELS.parent
    frames = [f for f in labels["frames"] if "expect" in f][: args.frames]
    results: dict = {"machine": "Ryzen 5 3600 CPU; all engines on CPU", "frames": len(frames), "engines": {}}

    # ---- mpl-ocr itself: same frames, same ROIs.
    ours: dict[str, list[float]] = {"hp": [], "mp": [], "exp": [], "map_name": []}
    ours_ok = {k: 0 for k in ours}
    rois_by_frame = {}
    for f in frames:
        path = base / f["image"]
        d = mapleocr_json(path, list(ours))
        rois_by_frame[f["image"]] = d["calibration"]["rois"]
        for r in d["readings"]:
            if r["roi"] not in ours:
                continue
            ours[r["roi"]].append(r["ms"])
            v = r["value"]
            if r["roi"] in ("hp", "mp") and v["type"] == "ratio":
                ours_ok[r["roi"]] += f"{v['current']}/{v['max']}" == f["expect"][r["roi"]]
            elif r["roi"] == "exp" and v["type"] == "exp":
                ours_ok["exp"] += f"{v['value']} ({v['percent']:.2f}%)" == f["expect"]["exp"]
            elif r["roi"] == "map_name" and v["type"] == "map":
                m = f.get("map_name") or {}
                ours_ok["map_name"] += (v["region"], v["name"]) == (m.get("region"), m.get("name"))
    results["mpl_ocr"] = {
        k: {"p50_ms": statistics.median(v), "p95_ms": pct(v, 0.95), "correct": ours_ok[k], "n": len(v)}
        for k, v in ours.items()
    }

    # ---- Python engines.
    for key in [k for k in args.engines.split(",") if k]:
        try:
            eng = {"rapid": Rapid, "tesseract": Tesseract, "easy": Easy}[key]()
        except Exception as e:  # noqa: BLE001
            print(f"skipping {key}: {e}")
            continue
        out: dict = {"name": eng.name}
        warm = np.asarray(Image.open(base / frames[0]["image"]).convert("RGB"))[1145:1170, 680:850]
        for _ in range(3):
            eng.read(warm)
        # ROI crops.
        per: dict[str, dict] = {k: {"ms": [], "ok": 0} for k in ("hp", "mp")}
        for f in frames:
            img = np.asarray(Image.open(base / f["image"]).convert("RGB"))
            for roi in ("hp", "mp"):
                r = rois_by_frame[f["image"]][roi]
                crop = img[r["y"] : r["y"] + r["h"], r["x"] : r["x"] + r["w"]]
                # Pad + 2x upscale: the usual advice for tiny UI text.
                crop = cv2.copyMakeBorder(crop, 12, 12, 12, 12, cv2.BORDER_REPLICATE)
                crop = cv2.resize(crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
                ms, text = timed(eng.read, crop)
                per[roi]["ms"].append(ms)
                per[roi]["ok"] += hp_ok(text, f["expect"][roi])
        out["roi"] = {
            k: {"p50_ms": statistics.median(v["ms"]), "p95_ms": pct(v["ms"], 0.95), "correct": v["ok"], "n": len(v["ms"])}
            for k, v in per.items()
        }
        # Full frame: HP anywhere on screen; tooltips.
        full_ms, hp_found, n = [], 0, 0
        for f in frames[: args.full_frames]:
            img = np.asarray(Image.open(base / f["image"]).convert("RGB"))
            ms, text = timed(eng.read, img)
            full_ms.append(ms)
            hp_found += hp_ok(text, f["expect"]["hp"])
            n += 1
        out["full_frame"] = {"p50_ms": statistics.median(full_ms), "p95_ms": pct(full_ms, 0.95), "hp_found": hp_found, "n": n}
        tt_found = tt_total = 0
        tt_ms = []
        for name, items in TOOLTIP_TRUTH.items():
            path = next((base / f["image"] for f in labels["frames"] if f["image"].endswith(name)), None)
            if path is None:
                continue
            ms, text = timed(eng.read, np.asarray(Image.open(path).convert("RGB")))
            tt_ms.append(ms)
            t = norm(text)
            for it in items:
                tt_total += 1
                tt_found += norm(it) in t
        out["tooltip_full_frame"] = {"p50_ms": statistics.median(tt_ms), "names_found": tt_found, "names_total": tt_total}
        results["engines"][key] = out
        print(f"{eng.name}: ROI hp p50 {out['roi']['hp']['p50_ms']:.1f} ms, full frame p50 {out['full_frame']['p50_ms']:.0f} ms")

    # ---- mpl-ocr tooltips (detection + read, full frame in).
    tt_ms, tt_found, tt_total = [], 0, 0
    for name, items in TOOLTIP_TRUTH.items():
        path = next((base / f["image"] for f in labels["frames"] if f["image"].endswith(name)), None)
        if path is None:
            continue
        d = mapleocr_json(path, ["tooltip"])
        r = d["readings"][0]
        tt_ms.append(r["ms"])
        # Name or any of its text lines (the other engines are scored on all text too).
        got = norm(" ".join(t["name"] + " " + " ".join(t["lines"]) for t in r["value"].get("items", [])))
        for it in items:
            tt_total += 1
            tt_found += norm(it) in got
    results["mpl_ocr"]["tooltip"] = {"p50_ms": statistics.median(tt_ms), "names_found": tt_found, "names_total": tt_total}

    out_path = ROOT / "bench" / "results.json"
    out_path.write_text(json.dumps(results, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()
