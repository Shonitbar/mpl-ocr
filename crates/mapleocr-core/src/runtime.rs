//! Live runtime: capture thread + recognition worker.

use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::Duration;

use arc_swap::{ArcSwap, ArcSwapOption};
use crossbeam_channel::{Receiver, Sender};
use mapleocr_capture::{BackendKind, Capture, Rect, TargetSelector, now_ticks};
use mapleocr_imgproc::{BgraImage, Gray, hash_pixels};
use mapleocr_layout::{AnchorTemplates, Calibration, Layout, calibrate};
use mapleocr_recog::Reading;

use crate::config::Config;
use crate::pipeline::{Grid, Packs, Pipeline};
use crate::{Error, Result, data_dir};

/// How long the capture thread blocks per wait; bounds shutdown latency only.
const WAIT: Duration = Duration::from_millis(100);
/// Cached readings per ROI before the cache is reset.
const CACHE_PER_ROI: usize = 256;

pub struct Options {
    /// Empty: the saved target from `mapleocr select`, else a "maple" heuristic.
    pub selector: TargetSelector,
    pub backend: Option<BackendKind>,
    pub layout: Layout,
    /// Subset of ROIs to run; `None` runs all.
    pub rois: Option<Vec<String>>,
    pub data_dir: PathBuf,
}

impl Options {
    /// Saved config, `<data_dir>/layout.yaml` if present (else the built-in
    /// layout), and every ROI.
    pub fn from_config() -> Result<Self> {
        let dir = data_dir()?;
        let user_layout = dir.join("layout.yaml");
        let layout = if user_layout.exists() { Layout::load(&user_layout)? } else { Layout::builtin() };
        Ok(Self { selector: TargetSelector::default(), backend: None, layout, rois: None, data_dir: dir })
    }
}

/// What the runtime is doing, for status displays.
#[derive(Clone, Debug, Default)]
pub struct Status {
    pub target: String,
    pub backend: &'static str,
    pub calibration: Option<Calibration>,
    /// Anchors were found by template match (vs. default positions).
    pub anchors_matched: bool,
    pub frames: u64,
    pub jobs: u64,
    pub cache_hits: u64,
    pub last_error: Option<String>,
}

pub type SubscriptionId = u64;
type Callback = Arc<dyn Fn(&Reading) + Send + Sync>;

struct Shared {
    names: Vec<Arc<str>>,
    latest: Vec<ArcSwapOption<Reading>>,
    subs: Mutex<Vec<(SubscriptionId, Vec<usize>, Callback)>>,
    next_sub: AtomicU64,
    running: AtomicBool,
    status: ArcSwap<Status>,
}

impl Shared {
    fn update_status(&self, f: impl FnOnce(&mut Status)) {
        let mut s = (**self.status.load()).clone();
        f(&mut s);
        self.status.store(Arc::new(s));
    }

    /// Store the latest reading; notify subscribers only if the value changed.
    fn publish(&self, i: usize, mut r: Reading) {
        r.published_at = now_ticks().0;
        let r = Arc::new(r);
        let changed = self.latest[i].load().as_ref().is_none_or(|prev| prev.value != r.value);
        self.latest[i].store(Some(r.clone()));
        if !changed {
            return;
        }
        let subs: Vec<Callback> = self
            .subs
            .lock()
            .expect("subscriber lock")
            .iter()
            .filter(|(_, rois, _)| rois.contains(&i))
            .map(|(_, _, f)| f.clone())
            .collect();
        for f in subs {
            f(&r);
        }
    }
}

struct Job {
    roi: usize,
    hash: u64,
    frame_id: u64,
    captured_at: i64,
    grid: Grid,
    image: BgraImage,
}

/// The running OCR framework. Dropping it stops capture and joins its threads.
pub struct MapleOcr {
    shared: Arc<Shared>,
    threads: Vec<JoinHandle<()>>,
}

impl MapleOcr {
    /// Find the game window (saved target or heuristic), calibrate on the
    /// first frame, and start reading every ROI of the layout.
    pub fn auto() -> Result<Self> {
        Self::start(Options::from_config()?)
    }

    pub fn start(opts: Options) -> Result<Self> {
        let cfg = Config::load()?;
        let selector = if opts.selector.is_empty() { cfg.target.clone().unwrap_or_default() } else { opts.selector };
        let backend = opts.backend.unwrap_or(cfg.backend);
        let packs = Packs::load(&opts.data_dir)?;
        let atlas_dir = opts.data_dir.join("atlas");
        let pipeline = Pipeline::new(&opts.layout, opts.rois.as_deref(), packs, Some(&atlas_dir))?;
        let names = pipeline.roi_names();
        let settle: Vec<u32> = (0..names.len()).map(|i| pipeline.settle_frames(i)).collect();

        let shared = Arc::new(Shared {
            latest: names.iter().map(|_| ArcSwapOption::empty()).collect(),
            names,
            subs: Mutex::new(Vec::new()),
            next_sub: AtomicU64::new(1),
            running: AtomicBool::new(true),
            status: ArcSwap::from_pointee(Status::default()),
        });
        let (tx, rx) = crossbeam_channel::unbounded::<Job>();

        let worker = {
            let shared = shared.clone();
            std::thread::Builder::new().name("mapleocr-recog".into()).spawn(move || worker(shared, pipeline, rx))?
        };

        // WGC/D3D objects are not Send: open capture on its own thread and
        // hand back only the outcome.
        let (opened_tx, opened_rx) = std::sync::mpsc::channel::<Result<()>>();
        let capture = {
            let shared = shared.clone();
            let layout = opts.layout.clone();
            let anchors_dir = opts.data_dir.join("anchors");
            std::thread::Builder::new().name("mapleocr-capture".into()).spawn(move || {
                let cap = match Capture::open(selector, backend) {
                    Ok(c) => c,
                    Err(e) => {
                        let _ = opened_tx.send(Err(e.into()));
                        return;
                    }
                };
                shared.update_status(|s| {
                    s.target = cap.target().describe();
                    s.backend = cap.backend_name();
                });
                let _ = opened_tx.send(Ok(()));
                capture_loop(&shared, cap, &layout, &anchors_dir, &settle, &tx);
            })?
        };
        let opened = opened_rx.recv().unwrap_or_else(|_| Err(Error::Config("capture thread died".into())));
        let me = Self { shared, threads: vec![capture, worker] };
        opened.map(|()| me)
    }

    pub fn roi_names(&self) -> &[Arc<str>] {
        &self.shared.names
    }

    fn index(&self, roi: &str) -> Result<usize> {
        self.shared.names.iter().position(|n| &**n == roi).ok_or_else(|| Error::UnknownRoi(roi.into()))
    }

    /// Latest reading for `roi`, without locking. `None` until the first one.
    pub fn read(&self, roi: &str) -> Result<Option<Arc<Reading>>> {
        Ok(self.shared.latest[self.index(roi)?].load_full())
    }

    /// Call `f` (on the recognition thread; keep it short) for every new
    /// reading of the given ROIs.
    pub fn subscribe(&self, rois: &[&str], f: impl Fn(&Reading) + Send + Sync + 'static) -> Result<SubscriptionId> {
        let idx = rois.iter().map(|r| self.index(r)).collect::<Result<Vec<_>>>()?;
        let id = self.shared.next_sub.fetch_add(1, Ordering::Relaxed);
        self.shared.subs.lock().expect("subscriber lock").push((id, idx, Arc::new(f)));
        Ok(id)
    }

    pub fn unsubscribe(&self, id: SubscriptionId) {
        self.shared.subs.lock().expect("subscriber lock").retain(|(i, _, _)| *i != id);
    }

    pub fn status(&self) -> Arc<Status> {
        self.shared.status.load_full()
    }

    /// Stop capture and recognition and wait for both threads.
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.shared.running.store(false, Ordering::Release);
        for t in self.threads.drain(..) {
            let _ = t.join();
        }
    }
}

impl Drop for MapleOcr {
    fn drop(&mut self) {
        self.shutdown();
    }
}

#[derive(Clone, Copy, Default)]
struct RoiState {
    last_hash: u64,
    stable: u32,
    sent_hash: Option<u64>,
}

fn calibrate_now(
    cap: &mut Capture,
    layout: &Layout,
    anchors_dir: &std::path::Path,
    shared: &Shared,
) -> Option<Calibration> {
    let (w, h) = cap.client_size();
    if cap.set_rois(&[Rect::new(0, 0, w, h)]).is_err() {
        return None;
    }
    let frame = loop {
        if !shared.running.load(Ordering::Acquire) {
            return None;
        }
        match cap.next_frame(WAIT) {
            Ok(Some(f)) => break f.rois[0].view().to_owned(),
            Ok(None) => continue,
            Err(e) => {
                shared.update_status(|s| s.last_error = Some(e.to_string()));
                return None;
            }
        }
    };
    let mut templates = AnchorTemplates::load(anchors_dir).ok().flatten();
    if templates.is_none() {
        // First run: cut anchor templates from this frame, if they have
        // enough texture to be real UI (not a loading screen).
        let boot = AnchorTemplates::bootstrap(layout, frame.view());
        if boot.images.values().all(textured) && boot.save(anchors_dir).is_ok() {
            templates = Some(boot);
        }
    }
    let calib = calibrate(layout, frame.view(), templates.as_ref());
    let matched = templates.is_some() && calib.all_matched();
    shared.update_status(|s| {
        s.calibration = Some(calib.clone());
        s.anchors_matched = matched;
    });
    Some(calib)
}

fn textured(g: &Gray) -> bool {
    let n = g.data.len().max(1) as f32;
    let mean = g.data.iter().sum::<f32>() / n;
    let var = g.data.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n;
    var.sqrt() > 12.0
}

fn capture_loop(
    shared: &Shared,
    mut cap: Capture,
    layout: &Layout,
    anchors_dir: &std::path::Path,
    settle: &[u32],
    tx: &Sender<Job>,
) {
    let mut calib: Option<Calibration> = None;
    let mut states = vec![RoiState::default(); shared.names.len()];
    let mut frames = 0u64;
    while shared.running.load(Ordering::Acquire) {
        if calib.is_none() {
            calib = calibrate_now(&mut cap, layout, anchors_dir, shared);
            let Some(c) = &calib else { continue };
            let rects: Vec<Rect> = shared.names.iter().map(|n| c.rois[&**n]).collect();
            if let Err(e) = cap.set_rois(&rects) {
                shared.update_status(|s| s.last_error = Some(e.to_string()));
                calib = None;
                continue;
            }
            states.fill(RoiState::default());
        }
        let frame = match cap.next_frame(WAIT) {
            Ok(Some(f)) => f,
            Ok(None) => continue,
            Err(e) => {
                shared.update_status(|s| s.last_error = Some(e.to_string()));
                std::thread::sleep(WAIT);
                continue;
            }
        };
        frames += 1;
        if calib.as_ref().is_some_and(|c| c.client_size != frame.client_size) {
            calib = None; // resized: recalibrate on the next frame
            continue;
        }
        let scale = calib.as_ref().map_or(1.0, |c| c.scale);
        for (i, (st, roi)) in states.iter_mut().zip(frame.rois).enumerate() {
            let h = hash_pixels(roi.data());
            if h != st.last_hash {
                st.last_hash = h;
                st.stable = 0;
            } else {
                st.stable = st.stable.saturating_add(1);
            }
            if st.stable == settle[i] && st.sent_hash != Some(h) {
                st.sent_hash = Some(h);
                let job = Job {
                    roi: i,
                    hash: h,
                    frame_id: frame.frame_id,
                    captured_at: frame.acquired.0,
                    grid: Grid { origin: (roi.rect().x, roi.rect().y), scale },
                    image: roi.view().to_owned(),
                };
                if tx.send(job).is_err() {
                    return;
                }
            }
        }
        if frames.is_multiple_of(60) {
            shared.update_status(|s| s.frames = frames);
        }
    }
}

fn worker(shared: Arc<Shared>, mut pipeline: Pipeline, rx: Receiver<Job>) {
    let mut cache: Vec<HashMap<u64, Reading>> = vec![HashMap::new(); shared.names.len()];
    let (mut jobs, mut hits) = (0u64, 0u64);
    let mut pending: Vec<Option<Job>> = (0..shared.names.len()).map(|_| None).collect();
    while shared.running.load(Ordering::Acquire) {
        let first = match rx.recv_timeout(WAIT) {
            Ok(j) => j,
            Err(crossbeam_channel::RecvTimeoutError::Timeout) => continue,
            Err(crossbeam_channel::RecvTimeoutError::Disconnected) => return,
        };
        // Coalesce: only the newest job per ROI matters.
        let roi = first.roi;
        pending[roi] = Some(first);
        while let Ok(j) = rx.try_recv() {
            let r = j.roi;
            pending[r] = Some(j);
        }
        for i in 0..pending.len() {
            let Some(job) = pending[i].take() else { continue };
            jobs += 1;
            if let Some(hit) = cache[i].get(&job.hash) {
                hits += 1;
                let r = Reading {
                    engine: "cache",
                    ms: 0.0,
                    frame_id: job.frame_id,
                    captured_at: job.captured_at,
                    ..hit.clone()
                };
                shared.publish(i, r);
                continue;
            }
            match pipeline.recognize(i, job.image.view(), job.grid, job.frame_id, job.captured_at) {
                Ok(r) => {
                    hits += u64::from(r.engine == "cache");
                    if cache[i].len() >= CACHE_PER_ROI {
                        cache[i].clear();
                    }
                    cache[i].insert(job.hash, r.clone());
                    shared.publish(i, r);
                }
                Err(e) => shared.update_status(|s| s.last_error = Some(format!("{}: {e}", shared.names[i]))),
            }
        }
        shared.update_status(|s| {
            s.jobs = jobs;
            s.cache_hits = hits;
        });
    }
}
