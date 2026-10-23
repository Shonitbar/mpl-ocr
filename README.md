# mpl-ocr

A low-latency, embeddable OCR framework that reads on-screen game state from
**MapleStory Classic World**: map name, HP/MP/EXP, damage numbers, inventory,
tooltips, chat, and more. Screen pixels in, structured readings out.

> **Unofficial.** Not affiliated with or endorsed by Nexon. Screen capture
> only: no memory reading, injection, or packet inspection. You are
> responsible for complying with the game's Terms of Service. See
> [DISCLAIMER.md](DISCLAIMER.md).

**Status: prototype.** Readable today: map name, level, job, character name,
HP, MP, EXP, NPC dialogue (name, text, options), quest helper (beta), chat
(beta), plus an opt-in whole-screen text scan. Tooltips, inventory and other
movable windows are next. See [docs/design.md](docs/design.md) for the
catalogue and measurements.

## Build

Requires Rust (stable, MSVC toolchain) on Windows 10 2004+ / Windows 11.

```sh
cargo install --path crates/mapleocr-cli --locked   # puts `mapleocr` in ~/.cargo/bin (on PATH)
```

Open a new terminal afterwards if Rust was installed in this session. Rerun
the command after pulling changes. Without installing, use
`.\target\release\mapleocr.exe` after `cargo build --release`.

## Try it

```sh
# 1. Find the game window and save it as the default target
mapleocr targets
mapleocr select --index 0            # or: --title MapleStory / --process MapleStory.exe

# 2. Read the map name live (calibrates on first run)
mapleocr calibrate
mapleocr watch map_name
mapleocr read map_name               # one JSON reading

# Optional: teach it your maps so readings snap to exact names
mapleocr pack add-map "Victoria Road" "Henesys"

# Teach the status-bar digit sprites (once; again when new digits appear).
# Type exactly what the ROI shows; the frame is grabbed from the game.
mapleocr atlas learn --roi hp --text "[356/598]"
mapleocr atlas learn --roi exp --text "10,105[91.53%]"
mapleocr atlas learn --roi level --text "17"

# Teach the NPC dialogue window's anchor from a saved frame where it is open
mapleocr calibrate --learn-anchor npc_dialogue --image fixtures/local/<session>/0010_full.png

# Everything at once, or the whole screen as text lines
mapleocr watch
mapleocr watch screen

# 3. Record samples: press F9 in game to save a frame, Ctrl+C to stop
mapleocr capture-dataset --label "inventory open"

# 4. Measure capture cost while the game is running
mapleocr bench capture --seconds 10
```

`capture-dataset` writes PNG + JSON sidecars to `fixtures/local/` (git-ignored).
Those images contain game pixels; keep them local.

Identify the UI font from labeled crops (Python tooling in `tools/`):

```sh
python -m venv tools/.venv && tools/.venv/Scripts/pip install -e "tools[dev]"
tools/.venv/Scripts/python -m mapleocr_tools.identify_font fixtures/local/font-samples.json
```

## Use it from Rust

```rust
use mapleocr_core::{MapleOcr, Value};

let ocr = MapleOcr::auto()?;            // find the game window, calibrate
ocr.subscribe(&["map_name"], |r| {
    if let Value::Map(m) = &r.value { println!("{} / {}", m.region, m.name) }
})?;
```

See [crates/mapleocr-core/examples/map_name.rs](crates/mapleocr-core/examples/map_name.rs).

## License

MIT. See [LICENSE](LICENSE).
