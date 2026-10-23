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
| `mapleocr-layout` | M1 | YAML ROI registry (anchor-relative), auto-calibration |
| `mapleocr-recog` | M2–M4 | glyph, sprite-digit, icon, CRNN, fallback OCR behind `Recognizer` |
| `mapleocr-semantic` | M5 | parsing + fuzzy snapping against data packs |
| `mapleocr-core` | M1+ | orchestration, dirty detection, cache, lock-free result ring |
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

## Font identification (M0, in progress)

Preliminary observations from one 1920×1200 capture, to be confirmed by
`identify-font` on a proper sample set:

- **Status bar numbers** (`HP[257/532]`, `MP[...]`, `EXP ...[..%]`) are
  sprite digits with fixed shapes, so a template/sprite matcher is ideal.
- **UI text** (map name, quest helper, party list, chat, system messages) is a
  **1-bit bitmap font** (no anti-aliasing, no ClearType fringes) that the
  client renders at a lower internal resolution and **upscales by a
  non-integer factor (~1.5×) with nearest-neighbour**. Evidence: 55–58% of
  adjacent columns in UI regions are pixel-identical vs. 7% in the game world,
  and mixed 1- and 2-pixel stroke widths, e.g. "Need" rendering as "Heed".
  Consequence: glyph templates vary with sub-pixel phase at output
  resolution, so the glyph matcher must first estimate the scale factor and
  phase per ROI and resample back to the native grid, then match exact 1-bit
  glyphs. `identify-font` measures this per ROI.
- Chat and system messages render over translucent or busy backgrounds, so they need
  color-keyed binarization before matching.

Per-ROI decision table: filled in after `identify-font` runs on the sample set.

| ROI | candidate engine | status |
|---|---|---|
| HP/MP/EXP numbers | sprite-digit matcher | likely |
| map name (minimap) | glyph matcher (native-grid) | likely |
| quest helper / party | glyph matcher | likely |
| chat / system log | glyph matcher + color key, CRNN fallback | open |
| tooltips / NPC dialogue | glyph matcher or CRNN | open |
| damage numbers | sprite-digit matcher | expected |
| inventory | icon matcher | expected |
