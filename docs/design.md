# mpl-ocr design

Living document. Numbers are measured, not assumed; each one names its setup.

## Goals and constraints

- Read MapleStory Classic World game state from **screen pixels only**: no
  memory reading, injection, hooking, or packet inspection.
- Priority order: latency, then accuracy, then resource use.
- Windows first. Core in Rust; training tooling in Python.
- No game assets in the repo. Atlases, icon DBs, data packs, and fixtures are
  generated locally from the user's own captures (`fixtures/local/`,
  `%LOCALAPPDATA%\mapleocr\`).

## Crates

| crate | status | role |
|---|---|---|
| `mapleocr-capture` | M0 | target discovery/selection, WGC + DXGI backends, GPU ROI crop |
| `mapleocr-cli` | M0 | `targets`, `select`, `capture-dataset`, `bench capture` |
| `mapleocr-imgproc` | M1 | image views, resampling, NCC template search, xxh3 ROI hashing |
| `mapleocr-layout` | M1 | YAML ROI registry (anchor-relative), auto-calibration |
| `mapleocr-recog` | M1–M2 | `Recognizer` trait, `Reading`, preprocessing, glyph matcher, Windows OCR; icon and CRNN later |
| `mapleocr-semantic` | M1 (maps) | parsing + fuzzy snapping against user-built data packs |
| `mapleocr-core` | M1 | `MapleOcr` runtime: capture thread, dirty detection, cache, recognition worker |
| `mapleocr-py`, `mapleocr-server` | M5 | PyO3 bindings, WebSocket + shared memory + C ABI |

Crates get added in the milestone that needs them. There are no empty stubs.

## Capture

### Target selection
The user picks the target while the game runs; nothing depends on the install
path.

- `mapleocr targets` lists Alt+Tab-visible windows (HWND, PID, exe, client
  size, DPI) and monitors.
- `mapleocr select --index N` (or `--title`, `--process`, `--pid`, `--hwnd`,
  `--monitor`) saves a `TargetSelector` to `%LOCALAPPDATA%\mapleocr\config.toml`.
  A window is saved as *process name + title*, not HWND, so a restarted client
  is picked up again.
- With nothing saved, a heuristic picks the largest window whose title or exe
  contains "maple".

### Backends
- **WGC** (`WgcBackend`): free-threaded frame pool; the `FrameArrived` callback
  signals an auto-reset Win32 event, and the consumer blocks on it, so there's no
  polling. Cursor capture is off (the cursor over an ROI would defeat dirty
  detection), and the capture border is turned off where the OS allows it. On wake, the
  backend drains to the newest frame. A content-size change recreates the pool and
  re-queries the client offset.
- **DXGI Desktop Duplication** (`DxgiBackend`): for exclusive fullscreen.
  Duplicates the game's monitor on the adapter that owns it and crops to the
  window's client rect. Mouse-only updates (`LastPresentTime == 0`) are skipped.
  `ACCESS_LOST` re-duplicates.
- **`Capture`** (the wrapper used by everything above): `Auto` opens WGC, and if no
  first frame arrives within 1.5 s and the window covers its monitor, it
  reopens on DXGI. If the target closes (`GraphicsCaptureItem.Closed` /
  `IsWindow`), it re-resolves the selector and rebinds with the same ROIs.
  Minimized windows simply produce no frames.
- The process is per-monitor-DPI-aware v2, so every coordinate is physical
  pixels (the dev machine runs at 120 DPI).

### GPU ROI crop
`RoiStager` owns one ROI-sized `STAGING` texture and one pre-allocated CPU
buffer per ROI. Per frame it queues all `CopySubresourceRegion`s, then maps
each, copies rows into the tight BGRA buffer, and unmaps. Parts of an ROI that
fall outside the frame are zeroed. Allocation-free unless the source format
changes.

### Timestamps
All timestamps are `Ticks` (100 ns) on the QPC timeline, the same clock as
WGC's `SystemRelativeTime`. Note: measured `present → acquired` is often
*negative* with WGC. `SystemRelativeTime` is the compositor's target vblank,
which lies slightly in the future when the frame is handed to us. Treat
`acquired → ready` as our capture cost. For true end-to-end latency, M1+ measures
`acquired → reading published`.

### M0 measurements
Setup: Ryzen 5 3600, RTX 3060 Ti, Windows 11, MapleStory client 1920×1200
windowed, release build.

| measurement | p50 | p95 | p99 |
|---|---|---|---|
| criterion, synthetic 1080p texture, 1 ROI 200×24 | 43 µs | | |
| criterion, 8 ROIs 200×24 | 104 µs | | |
| criterion, 32 ROIs 200×24 | 233 µs | | |
| criterion, full 1920×1080 frame | 1.81 ms | | |
| live WGC, 8 ROIs, `acquired → ready` | 0.46 ms | 1.96 ms | 2.78 ms |
| live WGC, full frame, `acquired → ready` | 2.20 ms | 2.84 ms | 3.74 ms |

Live, at about 57 fps delivered: **8 ROIs cost 6.2% of one core**, against 12.5% for
full-frame copies. The live p95 is dominated by `Map` waiting on a GPU that is
busy rendering the game. Candidate optimizations, to be measured before
they're adopted: double-buffered staging (map frame N-1 while N copies, which trades
one frame of latency, so probably not), a single atlas staging texture (one `Map`
instead of N), `MinUpdateInterval` to cap delivery rate per subscription.

## M1: map name end to end

```
capture thread                          recognition worker
  WGC frame ──► per-ROI xxh3 ──changed+settled──► raw-hash cache ─► preprocess (key, panel crop)
                 (unchanged: no work)              │                 └─► prepared-hash cache ─► WinOCR
                                                   └──────────────────────────────► parse + pack snap
                                                                  latest (ArcSwap, lock-free read) + subscribers (on change)
```

- **Layout** (`crates/mapleocr-layout/src/default_layout.yaml`): one anchor,
  the minimap's "MINI MAP" caption, and the `map_name` ROI relative to it, in
  native units (canvas 768 lines). The user can override it with
  `%LOCALAPPDATA%\mapleocr\layout.yaml`.
- **Calibration**: scale = client height / 768. On first run the anchor
  template is cut from the user's screen at its default position (only if
  it is textured, i.e. real UI) and saved to `%LOCALAPPDATA%\mapleocr\anchors`.
  Later runs locate it by coarse-to-fine NCC; below 0.75 the default position is
  used. The runtime recalibrates when the client size changes.
- **Dirty detection**: xxh3 per ROI per frame. A ROI is sent for recognition
  only when its hash changed and then stayed stable for `settle_frames` (2 for
  the map name, which fades during map changes).
- **Two caches**: one on the raw ROI hash, and one on the hash of the
  *prepared* image (color-keyed + cropped to the minimap panel). The minimap
  width varies, so the ROI also covers animated world pixels; those change the
  raw hash but not the prepared one, so they never reach OCR.
- **Fallback OCR = `Windows.Media.Ocr`**, built into Windows, instead of
  PaddleOCR/ONNX for M1: no model download, no extra runtime, and it measured
  100% on our fixtures at ~3 ms. It sits behind `Recognizer`, so the ONNX engine
  can replace or complement it when the CRNN tier lands (M4).
- **Preprocessing** (measured): white-text key, panel crop, no upscale, 16 px
  padding. With 8 px padding Windows OCR dropped short second lines ("Henesys"),
  and 2× upscale was slower with no gain.
- **Semantic**: two lines → `{region, name}`; snapped to
  `%LOCALAPPDATA%\mapleocr\packs\maps.json` (built with `mapleocr pack
  add-map`) at similarity ≥ 0.75. Snapped readings carry confidence = similarity;
  unchecked ones carry 0.5.
- **Notifications** fire only when the value changes; `read()` always returns the
  latest (lock-free `ArcSwap`).
- The capture thread makes itself per-monitor DPI aware, so embedding apps don't
  have to. The example caught this: without it, a 125%-scaled desktop gave
  wrong ROIs.

### M1 measurements
Setup as in M0 (Ryzen 5 3600, 1920×1200 windowed, release build).

| measurement | result |
|---|---|
| `bench fixtures`: 20 labeled frames, 4 distinct maps | **100% exact** (region + name), with or without pack |
| OCR recognize time (WinOCR, incl. preprocessing) | p50 2.9 ms, p95 3.4 ms; first call ~14 ms (engine warm-up) |
| live: capture → reading published (fresh OCR) | 3.6–4.4 ms (plus 2 settle frames, ≈33 ms after the change) |
| live: cache hit → published | 0.4 ms |
| live 30 s, map_name only, ~56 fps | 1680 frames, 88 jobs, 87 cache hits, **CPU 3.95% of one core** |

Known gaps (prototype): anchor-loss detection (only resize triggers
recalibration), job images are allocated per change (not pooled), and the
results ring buffer and shared-memory feed are M5. The 20 fixtures come from one
session and four maps, so accuracy will need a broader set.

## M2: the readable UI (glyph tier, windows, catalogue)

### What is readable today

| ROI | engine | value | status (16 held-out frames + live) |
|---|---|---|---|
| `map_name` | Windows OCR + map pack | `{region, name}` | 100% |
| `level` | glyph, learned `status_level` | number | 100%; **new digits must be taught** |
| `job`, `character_name` | glyph, GDI Arial 12 | text | 100% |
| `hp`, `mp` | glyph, learned `status_digits` | `{current, max}` | 100% |
| `exp` | glyph, learned `status_digits` | `{value, percent}` | 100% (incl. `10,105`) |
| `npc_name` | glyph, GDI Arial 11, `band` crop | text | 100% (2 open frames) |
| `npc_dialogue` | glyph, GDI Arial 12, presence anchor | `{text, options}` | 100% (2 open frames) |
| `quest_helper` | glyph, GDI Arial 11 + 11 bold | lines | **beta**: regular lines good; bold titles break on dark scenery |
| `chat` | Windows OCR on raw pixels | lines | **beta**: readable, noisy on busy backgrounds |
| `screen` (opt-in) | Windows OCR, whole client | lines with boxes | universal fallback, 230–270 ms |

Not yet: item tooltips (follow the cursor), inventory/equipment/stat/skill
windows (movable), damage numbers, shops. See "Next" below.

### Glyph tier
Fixed-font UI text is read on the **native grid**:

1. Color-key the ROI (`luma`, `dark`, `white`, `#rrggbb`, `raw`).
2. Area-resample it back by the UI scale (×1.5625 at 1200 px). The sub-pixel
   grid phase is found once per ROI by maximizing variance over 16 quarter-pixel
   phases, then cached until the geometry changes. Without it, 1-px strokes fall
   between native pixels and 5/9 and 6/8 become indistinguishable.
3. Binarize halfway between background and ink. 0.4 merged neighbouring
   digits; 0.5 keeps them apart and keeps 1-px dots.
4. Split into lines (empty rows) and glyphs (empty columns), after erasing
   dashed underlines. A dashed underline is a 1-px-tall row of equal runs at an
   equal pitch; monospaced digits also have regular runs, but they're stacked,
   so they're kept.
5. Match each glyph by IoU with ±1 px tolerance. Two-segment merges handle the
   game's notched bold `n`/`m`/`h`. A DP split handles touching letters, using
   pair glyphs (`ft`, `tt`, …) rendered by GDI.

**Atlases** are built locally:
- `gdi:<face>:<px>[:bold]`: rendered 1-bit through GDI from the user's
  system font, plus every letter pair GDI draws touching. Accept IoU ≥ 0.70.
- `learned:<name>`: sprite glyphs cut from the user's own captures with
  `mapleocr atlas learn` (`--image` or live). Accept IoU ≥ 0.85, so an unlearned
  `8` can't pass as a `5`. Each character may hold variants (sub-pixel phase
  differs by position). Learning refuses lines whose labels look shifted (a
  segment already matches a different character at ≥ 0.95) and size-inconsistent
  variants.
- A number containing an unknown glyph is reported as text with confidence
  0.2, **never as a guessed number**.

### Optional windows (presence anchors)
An anchor with `optional: true` (the NPC dialogue's left border) is checked
before its window is read. The capture thread copies only the anchor strip,
a few hundred pixels: it tests the exact expected position first, falls back to
a ±3 px search, and re-checks a closed window every 4th frame. While the window is
closed its ROIs report `Hidden`, and their pixels never leave the GPU. Their
templates are learned from a saved frame with `mapleocr calibrate --learn-anchor`.

### Content-aware crops
- `panel`: stop where the panel color ends (minimap width varies).
- `band:#rrggbb`: keep the tallest band of rows dominated by a color (the NPC
  name plate's height depends on the portrait).

### CPU: what it cost and what fixed it
Measured with `mapleocr watch` (which prints a per-thread breakdown) at
~54–56 fps, all default ROIs:

| change | CPU, % of one core |
|---|---|
| first live run, all ROIs | 73 |
| glyph cache on binarized-bitmap hash; rate limits (chat 300 ms, quest 500 ms) | 35–39 |
| presence on the capture thread; strict learned threshold | 29–33 |
| copy only what is needed: closed windows and not-yet-due ROIs stay on the GPU | 27–32 |
| phase cached per ROI (quest helper 24–47 → 3–5 ms) | 19–21 |
| `screen` really opt-in (it was copying + hashing the full frame every frame) | 8.7–10.4 |
| `Map` polls without waiting, yields ≤ 1 ms, then sleeps (staging p50 0.48 ms) | **11.6–12.5** |

Blocking `Map` made the NVIDIA driver busy-wait while the GPU rendered the
game. Pure sleeping cut CPU most but raised staging latency to 2.5 ms p50.
Latency comes first, so the yield window stays. Remaining cost is mostly chat
OCR (~14 ms per read, ≤ 3.3 Hz). The 10%-of-a-core target is slightly exceeded
with every ROI on.

### Next
- **Tooltips**: they follow the cursor, so they need a detector. Plan:
  border/background color search on a downscaled frame (GPU mip), then glyph
  reading inside with GDI Arial 12 (tooltip text was identified as Arial 12).
- **Movable windows** (inventory, equipment, stats, skills, quest log):
  title-bar templates searched at low rate on the downscaled frame, then
  window-relative ROIs. Inventory slots use the icon matcher.
- **Damage numbers**: sprite matcher with color classes and tracking.
- **Quest helper bold on dark scenery**: learn the bold glyphs from captures
  (`learned:`) instead of GDI, or key on the panel tint.
- **Chat**: glyph tier per chat color instead of OCR (lower CPU, higher accuracy).

## Font identification (M0)

Tool: `tools/mapleocr_tools/identify_font.py` (Python, offline; see "Tooling"
below). Input is labeled text crops from `capture-dataset` frames. Method: render
each candidate word with **GDI, non-antialiased** (how a classic Win32 client
draws text), upscale by a candidate factor, take the best normalized
cross-correlation (NCC) against the capture, and score per word so small
advance-width differences don't accumulate across a line. GDI matters: Pillow/FreeType
hint 11–14 px differently, and with FreeType the best line score was only 0.70.

Data: 20 frames, windowed 1920×1200 at 120 DPI, one session, 8 labeled text
samples plus 2 status-bar samples.

### Findings

1. **UI scale is uniform ×1.5625 = client height / 768**, estimated at ×1.560.
   The client draws its UI on a 768-line canvas and the window stretches it with
   a **smooth filter** (bicubic/bilinear; nearest-neighbour scores clearly
   worse: 0.89 vs 0.96). The dark undershoot halos around strokes confirm a
   bicubic-type kernel. *An earlier note here claimed nearest-neighbour
   upscaling. That was wrong.*
   At ×25/16 the sub-pixel phase pattern repeats every 16 native = 25 screen
   pixels.
2. **The source text is 1-bit (non-antialiased) GDI Arial.** For every
   identified ROI, the non-AA render beats grayscale-AA and ClearType renders.

| ROI | font (native px) | NCC | margin to next family | verdict |
|---|---|---|---|---|
| item tooltip body | Arial 12 regular | 0.961 | 0.146 | identified |
| NPC dialogue + options | Arial 12 regular | 0.905 | 0.221 | identified |
| chat (incl. yellow system lines) | Arial 14 regular | 0.875 | 0.207 | probable |
| quest helper lines | Arial 11 regular | 0.867 | 0.090 | probable |
| quest helper titles | Arial 11 bold | 0.871 | 0.029 | probable |
| minimap map name | none (best Tahoma 11 bold) | 0.653 | 0.003 | unresolved |
| HP/MP/EXP numbers | none | 0.448 | n/a | unresolved: sprites |

"Probable" rows score lower because their crops sit on gradients or
translucent backgrounds; the font ranking itself is stable.

3. Text over translucent or busy backgrounds (chat log, `[Notice]` lines)
   **must be color-keyed** before matching. The yellow notice line scores 0.92
   keyed vs 0.37 on plain luma.
4. The map name is white with a drop shadow and doesn't match any installed
   font. It may be a stylized render or a font that isn't installed (Gulim/Dotum
   are absent on the dev machine). The client process is protected, so its
   install path can't be read from the process; `--client-dir` lets a user
   point the tool at their own install to try any font files there.

### Decisions

| ROI | engine | why |
|---|---|---|
| tooltip, NPC dialogue, quest helper, chat | **glyph matcher** | 1-bit Arial at known sizes. Templates are generated **locally on the user's machine** from the system's Arial via GDI, then upscaled with the measured kernel per sub-pixel phase (16 phases at ×25/16). Nothing is shipped. CRNN stays as the fallback for low confidence. |
| chat | glyph matcher **+ per-channel color key** (chat type colors) | translucent background |
| map name | **glyph atlas built from captures** (`build-atlas`) + data-pack snapping | font unresolved, but the set of map names is finite |
| HP/MP/EXP numbers | **sprite-digit matcher**, atlas from captures | sprite digits, ~12 symbols |
| damage numbers | sprite-digit matcher | visibly sprite-rendered (frames 15, 18–20) |
| inventory / item slots | icon matcher | grids of icons |

Open: confirm the scale model (`client height / 768`) at a second window size and
in fullscreen. The current data is all 1920×1200 windowed.

## Tooling

Offline analysis and training tools are Python, in `tools/` (`pip install -e
tools[dev]`; ruff + pyright + pytest). `identify-font` lives there instead of in
`mapleocr-cli` because it needs GDI rendering and array maths, never runs on
the hot path, and feeds the Python training pipeline.

```
python -m mapleocr_tools.identify_font fixtures/local/font-samples.json --out report.json
```
Samples and reports stay in `fixtures/local/` (git-ignored) because they contain
game pixels and text.
