use std::sync::OnceLock;

use windows::Win32::System::Performance::{QueryPerformanceCounter, QueryPerformanceFrequency};

/// A point on the system QPC timeline in 100 ns units. This is the same clock
/// and unit as WGC's `SystemRelativeTime`, so timestamps from every backend
/// and from consumers compare directly.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Ticks(pub i64);

impl Ticks {
    pub fn micros_since(self, earlier: Ticks) -> f64 {
        (self.0 - earlier.0) as f64 / 10.0
    }
}

fn qpc_frequency() -> i64 {
    static FREQ: OnceLock<i64> = OnceLock::new();
    *FREQ.get_or_init(|| {
        let mut f = 0;
        // SAFETY: valid out-pointer; cannot fail on XP and later.
        unsafe { QueryPerformanceFrequency(&mut f) }.expect("QueryPerformanceFrequency");
        f
    })
}

/// Convert raw QPC counts (e.g. `DXGI_OUTDUPL_FRAME_INFO::LastPresentTime`).
pub(crate) fn ticks_from_qpc(qpc: i64) -> Ticks {
    Ticks((qpc as i128 * 10_000_000 / qpc_frequency() as i128) as i64)
}

pub fn now_ticks() -> Ticks {
    let mut c = 0;
    // SAFETY: valid out-pointer; cannot fail on XP and later.
    unsafe { QueryPerformanceCounter(&mut c) }.expect("QueryPerformanceCounter");
    ticks_from_qpc(c)
}
