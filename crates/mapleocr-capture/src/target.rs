//! Discovering and selecting what to capture: a top-level window or a monitor.

use std::ffi::c_void;

use serde::{Deserialize, Serialize};
use windows::Win32::Foundation::{CloseHandle, HWND, LPARAM, POINT, RECT};
use windows::Win32::Graphics::Dwm::{DWMWA_CLOAKED, DWMWA_EXTENDED_FRAME_BOUNDS, DwmGetWindowAttribute};
use windows::Win32::Graphics::Gdi::{
    ClientToScreen, EnumDisplayMonitors, GetMonitorInfoW, HDC, HMONITOR, MONITOR_DEFAULTTONEAREST, MONITORINFO,
    MONITORINFOEXW, MonitorFromWindow,
};
use windows::Win32::System::Threading::{
    OpenProcess, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION, QueryFullProcessImageNameW,
};
use windows::Win32::UI::HiDpi::{
    DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2, GetDpiForWindow, SetProcessDpiAwarenessContext,
    SetThreadDpiAwarenessContext,
};
use windows::Win32::UI::WindowsAndMessaging::{
    EnumWindows, GWL_EXSTYLE, GetClientRect, GetWindowLongPtrW, GetWindowTextLengthW, GetWindowTextW,
    GetWindowThreadProcessId, IsIconic, IsWindow, IsWindowVisible, WS_EX_TOOLWINDOW,
};
use windows::core::{BOOL, PWSTR};

use crate::{CaptureError, Result};

/// Make client-rect and capture coordinates physical pixels. Call once at
/// process start, before creating any window or capture.
pub fn init_dpi_awareness() {
    // SAFETY: plain Win32 call. It fails harmlessly if awareness was already set.
    let _ = unsafe { SetProcessDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) };
}

/// Make the calling thread per-monitor DPI aware, so window geometry is in
/// physical pixels (matching captured frames) without touching the rest of
/// the host process. [`crate::Capture::open`] calls this for its thread.
pub fn init_thread_dpi_awareness() {
    // SAFETY: plain Win32 call affecting only the calling thread.
    unsafe { SetThreadDpiAwarenessContext(DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2) };
}

#[derive(Clone, Debug, Serialize)]
pub struct WindowInfo {
    /// Raw `HWND` value.
    pub hwnd: isize,
    pub pid: u32,
    /// Executable file name, e.g. `MapleStory.exe`. Empty if access is denied.
    pub process: String,
    pub title: String,
    pub client_size: (u32, u32),
    pub dpi: u32,
    pub minimized: bool,
}

impl WindowInfo {
    pub fn hwnd(&self) -> HWND {
        HWND(self.hwnd as *mut c_void)
    }
}

#[derive(Clone, Debug, Serialize)]
pub struct MonitorInfo {
    pub index: usize,
    /// Raw `HMONITOR` value.
    pub handle: isize,
    pub name: String,
    /// Desktop coordinates `(left, top, right, bottom)`.
    pub bounds: (i32, i32, i32, i32),
    pub primary: bool,
}

impl MonitorInfo {
    pub fn hmonitor(&self) -> HMONITOR {
        HMONITOR(self.handle as *mut c_void)
    }

    pub fn size(&self) -> (u32, u32) {
        ((self.bounds.2 - self.bounds.0) as u32, (self.bounds.3 - self.bounds.1) as u32)
    }
}

#[derive(Clone, Debug)]
pub enum CaptureTarget {
    Window(WindowInfo),
    Monitor(MonitorInfo),
}

impl CaptureTarget {
    pub fn describe(&self) -> String {
        match self {
            Self::Window(w) => format!("window \"{}\" ({}, pid {})", w.title, w.process, w.pid),
            Self::Monitor(m) => format!("monitor {} ({})", m.index, m.name),
        }
    }
}

/// How the user chose the target. Saved to config so a recreated game window
/// (new HWND, new PID) is found again by process name and title.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct TargetSelector {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub hwnd: Option<isize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub pid: Option<u32>,
    /// Case-insensitive exact executable name, e.g. `msw.exe`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub process: Option<String>,
    /// Case-insensitive substring of the window title.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub title: Option<String>,
    /// Monitor index from [`list_monitors`]. Takes precedence over window fields.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub monitor: Option<usize>,
}

impl TargetSelector {
    pub fn is_empty(&self) -> bool {
        *self == Self::default()
    }

    /// Selector that would find this window again after it is recreated.
    pub fn for_window(w: &WindowInfo) -> Self {
        Self {
            process: (!w.process.is_empty()).then(|| w.process.clone()),
            title: Some(w.title.clone()),
            ..Self::default()
        }
    }

    fn matches(&self, w: &WindowInfo) -> bool {
        self.hwnd.is_none_or(|h| h == w.hwnd)
            && self.pid.is_none_or(|p| p == w.pid)
            && self.process.as_deref().is_none_or(|p| w.process.eq_ignore_ascii_case(p))
            && self.title.as_deref().is_none_or(|t| w.title.to_lowercase().contains(&t.to_lowercase()))
    }

    /// Resolve to a live target. An empty selector falls back to a
    /// "maple" heuristic over window titles and process names. Among several
    /// matching windows the one with the largest client area wins.
    pub fn resolve(&self) -> Result<CaptureTarget> {
        if let Some(i) = self.monitor {
            return list_monitors()
                .into_iter()
                .find(|m| m.index == i)
                .map(CaptureTarget::Monitor)
                .ok_or_else(|| CaptureError::NoMatch(format!("monitor index {i}")));
        }
        let windows = list_windows();
        let best = if self.is_empty() {
            // A "maple" process beats a "maple" title: browser tabs about the
            // game have it in their title too.
            windows
                .into_iter()
                .filter_map(|w| {
                    let by_process = w.process.to_lowercase().contains("maple");
                    let by_title = w.title.to_lowercase().contains("maple");
                    (by_process || by_title).then(|| ((by_process, w.client_size.0 as u64 * w.client_size.1 as u64), w))
                })
                .max_by_key(|(rank, _)| *rank)
                .map(|(_, w)| w)
        } else {
            windows
                .into_iter()
                .filter(|w| self.matches(w))
                .max_by_key(|w| w.client_size.0 as u64 * w.client_size.1 as u64)
        };
        best.map(CaptureTarget::Window).ok_or_else(|| CaptureError::NoMatch(format!("{self:?}")))
    }
}

/// Visible, non-tool, non-cloaked top-level windows with a title, i.e. the
/// windows a user would recognize in Alt+Tab.
pub fn list_windows() -> Vec<WindowInfo> {
    unsafe extern "system" fn cb(hwnd: HWND, lparam: LPARAM) -> BOOL {
        // SAFETY: lparam is the &mut Vec passed below, alive for the whole enumeration.
        let out = unsafe { &mut *(lparam.0 as *mut Vec<WindowInfo>) };
        if let Some(info) = window_info(hwnd) {
            out.push(info);
        }
        true.into()
    }
    let mut out = Vec::new();
    // SAFETY: the callback only touches `out` through lparam during this call.
    let _ = unsafe { EnumWindows(Some(cb), LPARAM(&mut out as *mut _ as isize)) };
    out
}

pub(crate) fn window_info(hwnd: HWND) -> Option<WindowInfo> {
    // SAFETY: all calls take a possibly-stale HWND and fail gracefully on one.
    unsafe {
        if !IsWindowVisible(hwnd).as_bool() {
            return None;
        }
        if GetWindowLongPtrW(hwnd, GWL_EXSTYLE) as u32 & WS_EX_TOOLWINDOW.0 != 0 {
            return None;
        }
        let mut cloaked = 0u32;
        if DwmGetWindowAttribute(hwnd, DWMWA_CLOAKED, &mut cloaked as *mut _ as *mut c_void, size_of::<u32>() as u32)
            .is_ok()
            && cloaked != 0
        {
            return None;
        }
        let len = GetWindowTextLengthW(hwnd);
        if len <= 0 {
            return None;
        }
        let mut buf = vec![0u16; len as usize + 1];
        let n = GetWindowTextW(hwnd, &mut buf);
        let title = String::from_utf16_lossy(&buf[..n as usize]);

        let mut pid = 0;
        GetWindowThreadProcessId(hwnd, Some(&mut pid));
        let mut rc = RECT::default();
        GetClientRect(hwnd, &mut rc).ok()?;
        Some(WindowInfo {
            hwnd: hwnd.0 as isize,
            pid,
            process: process_name(pid).unwrap_or_default(),
            title,
            client_size: ((rc.right - rc.left) as u32, (rc.bottom - rc.top) as u32),
            dpi: GetDpiForWindow(hwnd),
            minimized: IsIconic(hwnd).as_bool(),
        })
    }
}

fn process_name(pid: u32) -> Option<String> {
    // SAFETY: handle is closed before returning; buffer length is passed in `len`.
    unsafe {
        let h = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid).ok()?;
        let mut buf = [0u16; 1024];
        let mut len = buf.len() as u32;
        let r = QueryFullProcessImageNameW(h, PROCESS_NAME_WIN32, PWSTR(buf.as_mut_ptr()), &mut len);
        let _ = CloseHandle(h);
        r.ok()?;
        let path = String::from_utf16_lossy(&buf[..len as usize]);
        path.rsplit('\\').next().map(str::to_owned)
    }
}

pub fn list_monitors() -> Vec<MonitorInfo> {
    unsafe extern "system" fn cb(hmon: HMONITOR, _: HDC, _: *mut RECT, lparam: LPARAM) -> BOOL {
        // SAFETY: lparam is the &mut Vec passed below, alive for the whole enumeration.
        let out = unsafe { &mut *(lparam.0 as *mut Vec<MonitorInfo>) };
        let mut mi = MONITORINFOEXW {
            monitorInfo: MONITORINFO { cbSize: size_of::<MONITORINFOEXW>() as u32, ..Default::default() },
            ..Default::default()
        };
        // SAFETY: cbSize says the buffer is a MONITORINFOEXW.
        if unsafe { GetMonitorInfoW(hmon, &mut mi.monitorInfo) }.as_bool() {
            let r = mi.monitorInfo.rcMonitor;
            let name_len = mi.szDevice.iter().position(|&c| c == 0).unwrap_or(mi.szDevice.len());
            out.push(MonitorInfo {
                index: out.len(),
                handle: hmon.0 as isize,
                name: String::from_utf16_lossy(&mi.szDevice[..name_len]),
                bounds: (r.left, r.top, r.right, r.bottom),
                primary: mi.monitorInfo.dwFlags & 1 != 0, // MONITORINFOF_PRIMARY
            });
        }
        true.into()
    }
    let mut out = Vec::new();
    // SAFETY: the callback only touches `out` through lparam during this call.
    let _ = unsafe { EnumDisplayMonitors(None, None, Some(cb), LPARAM(&mut out as *mut _ as isize)) };
    out
}

/// Where a window's client area is, in screen pixels and relative to its
/// DWM frame (which is what WGC window capture delivers).
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct WindowGeometry {
    /// Client-area origin in screen coordinates.
    pub client_origin: (i32, i32),
    pub client_size: (u32, u32),
    /// Visible frame bounds (excluding the invisible resize border).
    pub frame: RECT,
    pub monitor: HMONITOR,
    pub minimized: bool,
}

impl WindowGeometry {
    pub fn query(hwnd: HWND) -> Result<Self> {
        // SAFETY: all calls validate the HWND; out-pointers are valid locals.
        unsafe {
            if !IsWindow(Some(hwnd)).as_bool() {
                return Err(CaptureError::TargetLost);
            }
            let mut rc = RECT::default();
            GetClientRect(hwnd, &mut rc)?;
            let mut origin = POINT::default();
            let _ = ClientToScreen(hwnd, &mut origin);
            let mut frame = RECT::default();
            DwmGetWindowAttribute(
                hwnd,
                DWMWA_EXTENDED_FRAME_BOUNDS,
                &mut frame as *mut _ as *mut c_void,
                size_of::<RECT>() as u32,
            )?;
            Ok(Self {
                client_origin: (origin.x, origin.y),
                client_size: ((rc.right - rc.left) as u32, (rc.bottom - rc.top) as u32),
                frame,
                monitor: MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST),
                minimized: IsIconic(hwnd).as_bool(),
            })
        }
    }

    /// Client origin relative to the WGC window texture.
    pub fn client_offset_in_frame(&self) -> (i32, i32) {
        (self.client_origin.0 - self.frame.left, self.client_origin.1 - self.frame.top)
    }

    /// True if the client area covers its whole monitor (exclusive or borderless fullscreen).
    pub fn covers_monitor(&self) -> bool {
        let mut mi = MONITORINFO { cbSize: size_of::<MONITORINFO>() as u32, ..Default::default() };
        // SAFETY: valid out-pointer with cbSize set.
        if !unsafe { GetMonitorInfoW(self.monitor, &mut mi) }.as_bool() {
            return false;
        }
        let m = mi.rcMonitor;
        self.client_origin == (m.left, m.top)
            && self.client_size == ((m.right - m.left) as u32, (m.bottom - m.top) as u32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn win(title: &str, process: &str, pid: u32) -> WindowInfo {
        WindowInfo {
            hwnd: 1,
            pid,
            process: process.into(),
            title: title.into(),
            client_size: (800, 600),
            dpi: 96,
            minimized: false,
        }
    }

    #[test]
    fn selector_matching_is_case_insensitive_and_conjunctive() {
        let w = win("MapleStory", "MapleStory.exe", 42);
        let s = TargetSelector { title: Some("maple".into()), ..Default::default() };
        assert!(s.matches(&w));
        let s = TargetSelector { process: Some("maplestory.EXE".into()), pid: Some(42), ..Default::default() };
        assert!(s.matches(&w));
        let s = TargetSelector { process: Some("maplestory.exe".into()), pid: Some(7), ..Default::default() };
        assert!(!s.matches(&w));
        // Process names match exactly, not by substring.
        let s = TargetSelector { process: Some("maple".into()), ..Default::default() };
        assert!(!s.matches(&w));
    }

    #[test]
    fn for_window_round_trips_through_toml_shape() {
        let s = TargetSelector::for_window(&win("MapleStory", "MapleStory.exe", 42));
        assert_eq!(s.process.as_deref(), Some("MapleStory.exe"));
        assert_eq!(s.title.as_deref(), Some("MapleStory"));
        assert!(s.hwnd.is_none() && s.pid.is_none());
    }
}
