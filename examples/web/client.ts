// Live readings in Node 22+ or any browser (both have a global WebSocket).
//   npx tsx client.ts hp mp exp
type Reading = {
  roi: string;
  value: { type: string; [k: string]: unknown };
  confidence: number;
  engine: string;
  ms: number;
};
type Message =
  | { type: "hello"; protocol: number; rois: string[] }
  | { type: "reading"; snapshot: boolean; reading: Reading }
  | { type: "error"; message: string };

const ws = new WebSocket("ws://127.0.0.1:8765/v1/stream");
const rois = process.argv.slice(2);

ws.onopen = () => {
  if (rois.length) ws.send(JSON.stringify({ op: "subscribe", rois }));
};
ws.onmessage = (e) => {
  const m: Message = JSON.parse(String(e.data));
  if (m.type === "reading") {
    const { roi, value, confidence } = m.reading;
    console.log(m.snapshot ? "now" : "new", roi, JSON.stringify(value), confidence.toFixed(2));
  } else if (m.type === "error") {
    console.error(m.message);
  }
};
