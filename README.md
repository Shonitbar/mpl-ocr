# mpl-ocr

A low-latency, embeddable OCR framework that reads on-screen game state from
**MapleStory Classic World**: map name, HP/MP/EXP, damage numbers, inventory,
tooltips, chat, and more. Screen pixels in, structured readings out.

> **Unofficial.** Not affiliated with or endorsed by Nexon. Screen capture
> only: no memory reading, injection, or packet inspection. You are
> responsible for complying with the game's Terms of Service. See
> [DISCLAIMER.md](DISCLAIMER.md).

**Status: milestone 0.** Capture, target selection, dataset capture and the
capture benchmark work. Recognizers come next. See [docs/design.md](docs/design.md).

## Build

Requires Rust (stable, MSVC toolchain) on Windows 10 2004+ / Windows 11.

```sh
cargo build --release
```

## Try it

```sh
# 1. Find the game window and save it as the default target
mapleocr targets
mapleocr select --index 0            # or: --title MapleStory / --process MapleStory.exe

# 2. Record samples: press F9 in game to save a frame, Ctrl+C to stop
mapleocr capture-dataset --label "inventory open"

# 3. Measure capture cost while the game is running
mapleocr bench capture --seconds 10
```

`capture-dataset` writes PNG + JSON sidecars to `fixtures/local/` (git-ignored).
Those images contain game pixels; keep them local.

Identify the UI font from labeled crops (Python tooling in `tools/`):

```sh
python -m venv tools/.venv && tools/.venv/Scripts/pip install -e "tools[dev]"
tools/.venv/Scripts/python -m mapleocr_tools.identify_font fixtures/local/font-samples.json
```

## License

MIT. See [LICENSE](LICENSE).
