# mpl-ocr server protocol (v1)

Start it with `mapleocr serve [rois...] [--port 8765] [--allow-origin <origin>]`.
It binds `127.0.0.1` only and refuses to bind anything else. A browser request
carrying an `Origin` header is rejected (403) unless the origin is
`localhost`/`127.0.0.1`/`[::1]` or was given with `--allow-origin`. Open
`http://127.0.0.1:8765/` for the built-in live dashboard.

## HTTP

| | |
|---|---|
| `GET /v1/status` | `{protocol, target, backend, rois[], client_size, ui_scale, anchors_matched, frames, recognition_jobs, cache_hits, last_error}` |
| `GET /v1/readings` | `{ "<roi>": Reading, ... }`, ROIs that have a reading |
| `GET /v1/read/<roi>` | one `Reading`, `null` if none yet, 404 for an unknown ROI |

## WebSocket `/v1/stream`

All frames are JSON text. The connection starts subscribed to every ROI.

Server to client:

```json
{"type":"hello","protocol":1,"rois":["hp","mp","..."]}
{"type":"reading","snapshot":false,"reading":{ ...Reading }}
{"type":"error","message":"..."}
```

`snapshot: true` marks the current value sent right after connecting or
subscribing; `false` marks a live change. Readings are sent only when the
value changes.

Client to server:

```json
{"op":"subscribe","rois":["hp","mp"]}     // replaces the set, replies with snapshots
{"op":"unsubscribe","rois":["mp"]}
{"op":"read","roi":"hp"}                  // one snapshot
```

## Reading

```json
{
  "roi": "hp",
  "value": {"type": "ratio", "current": 642, "max": 642},
  "confidence": 0.9,        // 0..1; below 0.6 means unverified
  "engine": "glyph",        // glyph | winocr | cache | presence | detector
  "ms": 0.25,               // recognition time
  "frame_id": 2,
  "captured_at": 198194445203,   // QPC, 100 ns units: same clock as published_at
  "published_at": 198195277412,
  "raw": "[642/642]",
  "snapped": false          // replaced by a data-pack entry
}
```

`(published_at - captured_at) / 10_000` is capture-to-publish latency in ms.

### Value types (`value.type`)

| type | fields |
|---|---|
| `number` | `value` |
| `ratio` | `current`, `max` (HP, MP) |
| `exp` | `value`, `percent` |
| `map` | `region`, `name` |
| `text` | `text` |
| `lines` | `lines[{text, rect}]` |
| `dialogue` | `text`, `options[]` |
| `tooltips` | `items[{name, lines[], rect}]` |
| `window` | `rect`, `fields{}`, `slots[{index, icon, name?, quantity?}]` |
| `damage` | `hits[{value?, text, rect}]` |
| `empty` | the ROI is visible but holds no text |
| `hidden` | the window or tooltip is not on screen |

A number containing a glyph the matcher cannot read comes back as `text`
with confidence 0.2, never as a guessed number.
