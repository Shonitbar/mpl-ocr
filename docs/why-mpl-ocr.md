# Why mpl-ocr?

mpl-ocr turns MapleStory Classic World's screen into structured data: HP, MP,
EXP, item tooltips, inventory, chat, NPC dialogue, damage numbers and more. It
works from screen pixels only (no memory reading, injection, or packet
inspection) and any language or web UI can consume it over a local WebSocket.

> Unofficial, not affiliated with Nexon. You are responsible for following the
> game's Terms of Service. See [DISCLAIMER.md](../DISCLAIMER.md).

## Speed: mpl-ocr vs. general-purpose Python OCR

Same saved frames, same machine (Ryzen 5 3600, CPU only), best case for the
Python engines (they get the exact HP crop, padded and upscaled 2x).

**Reading the HP number** (12 frames, median time per read):

| engine | time | correct |
|---|---|---|
| **mpl-ocr** | **0.25 ms** | **12 / 12** |
| EasyOCR (CPU) | 125 ms (**500x slower**) | 12 / 12 |
| Tesseract | 138 ms (**550x slower**) | 11 / 12 |
| RapidOCR (PaddleOCR on ONNX) | 284 ms (**1100x slower**) | 12 / 12 |

**Reading the MP number** (the digits are smaller, and the engines struggle):

| engine | correct |
|---|---|
| **mpl-ocr** | **12 / 12** |
| EasyOCR | 10 / 12 |
| RapidOCR | 8 / 12 |
| Tesseract | 3 / 12 |

**Item tooltips from a full frame** (3 frames, 5 item names; mpl-ocr finds each
tooltip by its title bullet, the Python engines read the whole screen):

| engine | time | names found |
|---|---|---|
| **mpl-ocr** | **~60-100 ms** (first read; cached while the tooltip stays up) | 4 / 5 |
| Tesseract | 1.1 s (11-17x slower) | 5 / 5 |
| RapidOCR | 2.3 s (23-38x slower) | 4 / 5 |
| EasyOCR (CPU) | 12.6 s (**125-200x slower**) | 5 / 5 |

Tooltip accuracy is **not yet where it needs to be**: against hand-labeled lines,
mpl-ocr gets 5 of 6 names and 15 of 25 text lines exactly right (the Python engines
were scored only on whether the name appears anywhere, a more lenient test).
The speed is real; the accuracy work is in progress (see
[docs/design.md](design.md)).

**Finding the HP number anywhere on a 1920x1200 frame** (5 frames): RapidOCR
3.7 s (5/5), Tesseract 1.1 s (0/5), EasyOCR 12.7 s (0/5). mpl-ocr does not
search: it calibrates once and reads the known region in 0.25 ms.

**Why it is this much faster:** it does not run a general text model. Fixed-font
UI text and sprite digits are matched against exact glyph shapes taught from
your own screen; unchanged regions are skipped by hash, so a still screen costs
almost nothing. Measured at ~56 fps on the same machine:

| what is running | CPU |
|---|---|
| status bar + map readers (HP, MP, EXP, level, job, name, map) | **~6% of one core** |
| everything: plus chat, quest helper, tooltip / inventory / damage scouting | ~19% of one core |

A Python OCR call on the HP crop alone takes 125-284 ms, so it cannot keep up
with a single region at 60 fps, let alone all of them.

## What you can build with it

Stable = accuracy checked on held-out frames. Beta = works, with known gaps.

| capability | status | typical use |
|---|---|---|
| **HP / MP** as numbers (`current`, `max`) | stable | overlays, stream widgets, health-based alerts |
| **EXP** value and percent | stable | EXP/hour trackers, leveling analytics |
| **Level, job, character name** | stable | multi-character dashboards |
| **Map name** (snaps to your own map list) | stable | OBS scene switching, per-map stats, location logs |
| **Item tooltips** (item name + stat lines, finds the box anywhere) | **beta**: 5/6 names, ~60% of lines exact | price-lookup overlays, item databases, loot logs |
| **Inventory window** (finds it wherever you drag it; item icons recognized once you name them) | stable for detection, you name the icons | inventory trackers, drop logs, meso/item history |
| **NPC dialogue** (name, text, selectable options; reads only while open) | stable on tested frames | dialogue translation, quest guides, text-to-speech |
| **Damage numbers** | beta (18 of 20 held-out numbers right) | damage meters, DPS logging |
| **Quest helper** text | beta | quest progress overlays |
| **Chat log** | beta | chat logging, translation |
| **Whole-screen text scan** (every line with its position) | works, ~250 ms, opt-in | anything without a dedicated reader |

Also: runs without touching the game process, a local-only server (loopback,
Origin-checked), a live dashboard page, Python and TypeScript examples, a
`MapleOcr::auto()` Rust API, and automatic calibration for your window size.

## How the numbers were measured

Reproduce: `pip install -r bench/requirements.txt`, then
`python bench/compare.py` (results in [bench/results.json](../bench/results.json)).

Caveats, stated plainly:

- One machine, one play session, 1920x1200 windowed. Small samples: 12 frames
  for HP/MP, 3 frames for tooltips. Treat the speed ratios as solid and the
  accuracy percentages as indicative.
- mpl-ocr's HP/MP/EXP digit sprites are **taught from your own screen** first
  (`mapleocr atlas learn`). 4 of the 12 frames above were used for teaching;
  on 16 frames never used for teaching, status bar fields were 100% correct,
  and across 150 frames of live play all fields read with no unknown glyph (8
  random frames checked by eye: 24 of 24 values right).
- The times are recognition time per read once the pixels are in memory.
  Screen capture itself costs about 0.5 ms for the needed regions, and is
  shared by all readers.
- The Python engines were not tuned beyond the usual advice (crop, upscale,
  invert and threshold for Tesseract). A model fine-tuned on this game's fonts
  would close the accuracy gap, but not the speed gap.
- Windows only for now (Windows Graphics Capture + the system OCR engine for
  chat and tooltip text).
