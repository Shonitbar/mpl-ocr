//! Global hotkey on a dedicated message-loop thread.

use std::sync::Arc;
use std::sync::atomic::{AtomicU32, Ordering};

use anyhow::{Result, anyhow, bail};
use windows::Win32::UI::Input::KeyboardAndMouse::{MOD_NOREPEAT, RegisterHotKey, VK_F1};
use windows::Win32::UI::WindowsAndMessaging::{GetMessageW, MSG, WM_HOTKEY};

/// Counts presses of a global hotkey. The listening thread blocks in
/// `GetMessageW`, so it costs nothing while idle.
pub struct Hotkey {
    presses: Arc<AtomicU32>,
    seen: u32,
}

/// Parse `F1`..`F24`.
fn parse_vk(name: &str) -> Result<u32> {
    let n: u32 = name
        .strip_prefix(['F', 'f'])
        .and_then(|d| d.parse().ok())
        .ok_or_else(|| anyhow!("hotkey must be F1..F24, got {name:?}"))?;
    if !(1..=24).contains(&n) {
        bail!("hotkey must be F1..F24, got {name:?}");
    }
    Ok(VK_F1.0 as u32 + n - 1)
}

impl Hotkey {
    pub fn spawn(name: &str) -> Result<Self> {
        let vk = parse_vk(name)?;
        let presses = Arc::new(AtomicU32::new(0));
        let (tx, rx) = std::sync::mpsc::channel();
        let counter = presses.clone();
        std::thread::Builder::new().name("hotkey".into()).spawn(move || {
            // SAFETY: registered on this thread (hwnd None), so WM_HOTKEY is
            // posted to this thread's queue, which we drain below.
            unsafe {
                if let Err(e) = RegisterHotKey(None, 1, MOD_NOREPEAT, vk) {
                    let _ = tx.send(Err(e));
                    return;
                }
                let _ = tx.send(Ok(()));
                let mut msg = MSG::default();
                while GetMessageW(&mut msg, None, 0, 0).as_bool() {
                    if msg.message == WM_HOTKEY {
                        counter.fetch_add(1, Ordering::Relaxed);
                    }
                }
            }
        })?;
        rx.recv()?.map_err(|e| anyhow!("could not register hotkey {name} (in use by another app?): {e}"))?;
        Ok(Self { presses, seen: 0 })
    }

    /// True once per press since the last call.
    pub fn take(&mut self) -> bool {
        let now = self.presses.load(Ordering::Relaxed);
        let pressed = now != self.seen;
        self.seen = now;
        pressed
    }
}
