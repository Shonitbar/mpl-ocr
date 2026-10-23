"""Print live readings from a running `mapleocr serve`.

    pip install websockets
    python client.py                 # everything
    python client.py hp mp exp       # only these
"""

import asyncio
import json
import sys

import websockets

URL = "ws://127.0.0.1:8765/v1/stream"


def describe(v: dict) -> str:
    t = v["type"]
    if t == "ratio":
        return f"{v['current']}/{v['max']}"
    if t == "exp":
        return f"{v['value']} ({v['percent']:.2f}%)"
    if t == "map":
        return f"{v['region']} / {v['name']}"
    if t == "number":
        return str(v["value"])
    if t == "text":
        return v["text"]
    return json.dumps(v)[:120]


async def main(rois: list[str]) -> None:
    async with websockets.connect(URL) as ws:
        hello = json.loads(await ws.recv())
        print("connected, protocol", hello["protocol"], "ROIs:", ", ".join(hello["rois"]))
        if rois:
            await ws.send(json.dumps({"op": "subscribe", "rois": rois}))
        async for raw in ws:
            msg = json.loads(raw)
            if msg["type"] == "reading":
                r = msg["reading"]
                tag = "now " if msg["snapshot"] else "new "
                print(f"{tag}{r['roi']:<16} {describe(r['value'])}   [{r['engine']}, {r['confidence']:.2f}]")
            elif msg["type"] == "error":
                print("error:", msg["message"])


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))
