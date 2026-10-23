# mpl-ocr

A low-latency, embeddable OCR framework that reads on-screen game state from
**MapleStory Classic World**: map name, HP/MP/EXP, damage numbers, inventory,
tooltips, chat, and more. Screen pixels in, structured readings out.

> **Unofficial.** Not affiliated with or endorsed by Nexon. Screen capture
> only: no memory reading, injection, or packet inspection. You are
> responsible for complying with the game's Terms of Service. See
> [DISCLAIMER.md](DISCLAIMER.md).

**Status: milestone 1 prototype.** The map name is read end to end: window
discovery, auto-calibration, dirty detection, OCR, and data-pack snapping.
See [docs/design.md](docs/design.md).

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
