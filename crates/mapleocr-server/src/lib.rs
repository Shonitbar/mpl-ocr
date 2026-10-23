//! Local HTTP + WebSocket server for mpl-ocr.
//!
//! Any language or web UI can read the framework's output:
//!
//! | endpoint | |
//! |---|---|
//! | `GET /` | built-in live dashboard |
//! | `GET /v1/status` | target, backend, calibration, timing |
//! | `GET /v1/readings` | latest reading of every ROI, as `{roi: reading}` |
//! | `GET /v1/read/{roi}` | latest reading of one ROI (404 if unknown, `null` if none yet) |
//! | `GET /v1/stream` | WebSocket, see [`PROTOCOL`] |
//!
//! The server binds loopback only. Browsers send an `Origin` header, which is
//! rejected unless it is a localhost origin or was passed to `allow_origins`,
//! so an arbitrary web page cannot read your game state.

use std::collections::HashSet;
use std::net::SocketAddr;
use std::sync::Arc;

use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Request, State};
use axum::http::{HeaderValue, StatusCode, header};
use axum::middleware::{self, Next};
use axum::response::{Html, IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use mapleocr_core::{MapleOcr, Reading};
use serde_json::{Value, json};
use tokio::sync::broadcast;

/// WebSocket protocol, version 1. All messages are JSON text frames.
///
/// Server to client:
/// - `{"type":"hello","protocol":1,"rois":["hp",...]}` on connect. The
///   connection starts subscribed to every ROI.
/// - `{"type":"reading","snapshot":bool,"reading":{...}}` for each new
///   reading (`snapshot: true` for the current value sent on subscribe).
///   A reading has `roi`, `value` (tagged by `type`), `confidence`, `engine`,
///   `ms`, `frame_id`, `captured_at`, `published_at` (QPC, 100 ns units).
/// - `{"type":"error","message":"..."}`
///
/// Client to server:
/// - `{"op":"subscribe","rois":["hp"]}`: replaces the subscription set and
///   sends a snapshot of each ROI's latest reading.
/// - `{"op":"unsubscribe","rois":["hp"]}`
/// - `{"op":"read","roi":"hp"}`: one snapshot.
pub const PROTOCOL: u32 = 1;

const DASHBOARD: &str = include_str!("dashboard.html");

#[derive(Clone)]
struct App {
    ocr: Arc<MapleOcr>,
    tx: broadcast::Sender<Arc<Reading>>,
    allow_origins: Arc<Vec<String>>,
}

/// Run the server until Ctrl+C. Builds its own tokio runtime.
pub fn run_blocking(ocr: Arc<MapleOcr>, addr: SocketAddr, allow_origins: Vec<String>) -> anyhow::Result<()> {
    tokio::runtime::Builder::new_multi_thread().worker_threads(2).enable_all().build()?.block_on(serve(
        ocr,
        addr,
        allow_origins,
    ))
}

pub async fn serve(ocr: Arc<MapleOcr>, addr: SocketAddr, allow_origins: Vec<String>) -> anyhow::Result<()> {
    anyhow::ensure!(
        addr.ip().is_loopback(),
        "refusing to bind {addr}: the server exposes your screen's contents, loopback only"
    );
    let (tx, _) = broadcast::channel(1024);
    let names: Vec<String> = ocr.roi_names().iter().map(|n| n.to_string()).collect();
    let refs: Vec<&str> = names.iter().map(String::as_str).collect();
    let sender = tx.clone();
    ocr.subscribe(&refs, move |r| {
        // No receivers is fine: nobody is connected.
        let _ = sender.send(Arc::new(r.clone()));
    })?;
    let app = App { ocr, tx, allow_origins: Arc::new(allow_origins) };
    let router = Router::new()
        .route("/", get(|| async { Html(DASHBOARD) }))
        .route("/v1/status", get(status))
        .route("/v1/readings", get(readings))
        .route("/v1/read/{roi}", get(read_one))
        .route("/v1/stream", get(stream))
        .layer(middleware::from_fn_with_state(app.clone(), origin_guard))
        .with_state(app);
    let listener = tokio::net::TcpListener::bind(addr).await?;
    eprintln!("serving on http://{}  (WebSocket ws://{}/v1/stream)", listener.local_addr()?, listener.local_addr()?);
    axum::serve(listener, router)
        .with_graceful_shutdown(async {
            let _ = tokio::signal::ctrl_c().await;
        })
        .await?;
    Ok(())
}

fn origin_allowed(origin: &str, allow: &[String]) -> bool {
    let host = origin.split_once("://").map_or(origin, |(_, h)| h);
    let host = host.split('/').next().unwrap_or(host);
    let name = if let Some(rest) = host.strip_prefix('[') {
        rest.split(']').next().unwrap_or("")
    } else {
        host.split(':').next().unwrap_or("")
    };
    matches!(name, "localhost" | "127.0.0.1" | "::1") || allow.iter().any(|a| a == origin)
}

async fn origin_guard(State(app): State<App>, req: Request, next: Next) -> Response {
    let origin = req.headers().get(header::ORIGIN).and_then(|v| v.to_str().ok()).map(str::to_owned);
    if let Some(o) = &origin
        && !origin_allowed(o, &app.allow_origins)
    {
        return (StatusCode::FORBIDDEN, "origin not allowed; pass --allow-origin to permit it").into_response();
    }
    let mut res = next.run(req).await;
    if let Some(o) = origin
        && let Ok(v) = HeaderValue::from_str(&o)
    {
        res.headers_mut().insert(header::ACCESS_CONTROL_ALLOW_ORIGIN, v);
    }
    res
}

async fn status(State(app): State<App>) -> Json<Value> {
    let s = app.ocr.status();
    Json(json!({
        "protocol": PROTOCOL,
        "target": s.target,
        "backend": s.backend,
        "rois": app.ocr.roi_names(),
        "anchors_matched": s.anchors_matched,
        "client_size": s.calibration.as_ref().map(|c| c.client_size),
        "ui_scale": s.calibration.as_ref().map(|c| c.scale),
        "frames": s.frames,
        "recognition_jobs": s.jobs,
        "cache_hits": s.cache_hits,
        "gpu_staging_ms_total": s.stage_ms,
        "last_error": s.last_error,
    }))
}

async fn readings(State(app): State<App>) -> Json<Value> {
    let mut m = serde_json::Map::new();
    for n in app.ocr.roi_names() {
        if let Ok(Some(r)) = app.ocr.read(n) {
            m.insert(n.to_string(), serde_json::to_value(&*r).unwrap_or(Value::Null));
        }
    }
    Json(Value::Object(m))
}

async fn read_one(State(app): State<App>, Path(roi): Path<String>) -> Response {
    match app.ocr.read(&roi) {
        Ok(r) => Json(r.map(|r| serde_json::to_value(&*r).unwrap_or(Value::Null))).into_response(),
        Err(_) => (StatusCode::NOT_FOUND, Json(json!({"error": format!("unknown ROI {roi:?}")}))).into_response(),
    }
}

async fn stream(ws: WebSocketUpgrade, State(app): State<App>) -> Response {
    ws.on_upgrade(move |socket| session(socket, app))
}

fn reading_msg(r: &Reading, snapshot: bool) -> Message {
    Message::Text(json!({"type": "reading", "snapshot": snapshot, "reading": r}).to_string().into())
}

fn error_msg(message: &str) -> Message {
    Message::Text(json!({"type": "error", "message": message}).to_string().into())
}

async fn session(mut socket: WebSocket, app: App) {
    let names: Vec<String> = app.ocr.roi_names().iter().map(|n| n.to_string()).collect();
    let mut rx = app.tx.subscribe();
    let mut subs: HashSet<String> = names.iter().cloned().collect();
    let hello = json!({"type": "hello", "protocol": PROTOCOL, "rois": names});
    if socket.send(Message::Text(hello.to_string().into())).await.is_err() {
        return;
    }
    // Current values first, so a late joiner sees the state, not just changes.
    for n in &names {
        if let Ok(Some(r)) = app.ocr.read(n)
            && socket.send(reading_msg(&r, true)).await.is_err()
        {
            return;
        }
    }
    loop {
        tokio::select! {
            incoming = socket.recv() => {
                let Some(Ok(msg)) = incoming else { return };
                let Message::Text(text) = msg else { continue };
                let reply = handle_op(&app, &mut subs, &text);
                for m in reply {
                    if socket.send(m).await.is_err() {
                        return;
                    }
                }
            }
            event = rx.recv() => match event {
                Ok(r) if subs.contains(&*r.roi) => {
                    if socket.send(reading_msg(&r, false)).await.is_err() {
                        return;
                    }
                }
                Ok(_) | Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => return,
            },
        }
    }
}

fn handle_op(app: &App, subs: &mut HashSet<String>, text: &str) -> Vec<Message> {
    let Ok(v) = serde_json::from_str::<Value>(text) else { return vec![error_msg("invalid JSON")] };
    let rois = |v: &Value| -> Vec<String> {
        v["rois"]
            .as_array()
            .map(|a| a.iter().filter_map(|x| x.as_str().map(str::to_owned)).collect())
            .unwrap_or_default()
    };
    let known = |r: &str| app.ocr.roi_names().iter().any(|n| &**n == r);
    match v["op"].as_str() {
        Some("subscribe") => {
            let list = rois(&v);
            if let Some(bad) = list.iter().find(|r| !known(r)) {
                return vec![error_msg(&format!("unknown ROI {bad:?}"))];
            }
            *subs = list.iter().cloned().collect();
            list.iter().filter_map(|r| app.ocr.read(r).ok().flatten()).map(|r| reading_msg(&r, true)).collect()
        }
        Some("unsubscribe") => {
            for r in rois(&v) {
                subs.remove(&r);
            }
            Vec::new()
        }
        Some("read") => match v["roi"].as_str() {
            Some(r) if known(r) => {
                app.ocr.read(r).ok().flatten().map(|r| vec![reading_msg(&r, true)]).unwrap_or_default()
            }
            _ => vec![error_msg("read needs a known \"roi\"")],
        },
        _ => vec![error_msg("unknown op; use subscribe, unsubscribe or read")],
    }
}

#[cfg(test)]
mod tests {
    use super::origin_allowed;

    #[test]
    fn origin_policy() {
        assert!(origin_allowed("http://localhost:3000", &[]));
        assert!(origin_allowed("http://127.0.0.1:8765", &[]));
        assert!(origin_allowed("http://[::1]:8765", &[]));
        assert!(!origin_allowed("https://evil.example", &[]));
        assert!(!origin_allowed("http://localhost.evil.example", &[]));
        assert!(origin_allowed("https://app.example", &["https://app.example".into()]));
    }
}
