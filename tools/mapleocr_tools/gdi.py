"""Render text with Windows GDI, the way a classic Win32 game client would.

Pillow/FreeType hint small sizes differently from GDI, which matters at
1-bit 11-14 px. Rendering through GDI itself gives pixel-exact candidates.
"""

from __future__ import annotations

import ctypes as C
import sys
from ctypes import wintypes as W
from enum import IntEnum
from pathlib import Path

import numpy as np
import numpy.typing as npt

if sys.platform != "win32":  # pragma: no cover
    raise ImportError("mapleocr_tools.gdi requires Windows")

_gdi = C.windll.gdi32

_gdi.CreateFontW.restype = W.HFONT
_gdi.CreateFontW.argtypes = [C.c_int] * 5 + [W.DWORD] * 8 + [W.LPCWSTR]
_gdi.CreateCompatibleDC.restype = W.HDC
_gdi.CreateCompatibleDC.argtypes = [W.HDC]
_gdi.SelectObject.restype = W.HGDIOBJ
_gdi.SelectObject.argtypes = [W.HDC, W.HGDIOBJ]
_gdi.DeleteObject.argtypes = [W.HGDIOBJ]
_gdi.DeleteDC.argtypes = [W.HDC]
_gdi.TextOutW.argtypes = [W.HDC, C.c_int, C.c_int, W.LPCWSTR, C.c_int]
_gdi.SetTextColor.argtypes = [W.HDC, W.COLORREF]
_gdi.SetBkColor.argtypes = [W.HDC, W.COLORREF]
_gdi.GetTextExtentPoint32W.argtypes = [W.HDC, W.LPCWSTR, C.c_int, C.POINTER(W.SIZE)]
_gdi.CreateDIBSection.restype = W.HBITMAP
_gdi.CreateDIBSection.argtypes = [W.HDC, C.c_void_p, W.UINT, C.POINTER(C.c_void_p), W.HANDLE, W.DWORD]
_gdi.AddFontResourceExW.argtypes = [W.LPCWSTR, W.DWORD, C.c_void_p]
_gdi.AddFontResourceExW.restype = C.c_int


class Quality(IntEnum):
    """LOGFONT lfQuality values."""

    NONANTIALIASED = 3
    ANTIALIASED = 4
    CLEARTYPE = 5


class _BitmapInfoHeader(C.Structure):
    _fields_ = [
        ("biSize", W.DWORD),
        ("biWidth", W.LONG),
        ("biHeight", W.LONG),
        ("biPlanes", W.WORD),
        ("biBitCount", W.WORD),
        ("biCompression", W.DWORD),
        ("biSizeImage", W.DWORD),
        ("biXPelsPerMeter", W.LONG),
        ("biYPelsPerMeter", W.LONG),
        ("biClrUsed", W.DWORD),
        ("biClrImportant", W.DWORD),
    ]


def render(
    face: str,
    px: int,
    text: str,
    *,
    bold: bool = False,
    quality: Quality = Quality.NONANTIALIASED,
    pad: int = 1,
) -> npt.NDArray[np.uint8]:
    """White-on-black grayscale render of `text`.

    `px` is the character (em) height in pixels, i.e. LOGFONT lfHeight = -px.
    """
    dc = _gdi.CreateCompatibleDC(None)
    font = _gdi.CreateFontW(-px, 0, 0, 0, 700 if bold else 400, 0, 0, 0, 1, 0, 0, int(quality), 0, face)
    old_font = _gdi.SelectObject(dc, font)
    try:
        size = W.SIZE()
        _gdi.GetTextExtentPoint32W(dc, text, len(text), C.byref(size))
        w, h = max(size.cx, 1) + 2 * pad, max(size.cy, 1) + 2 * pad
        bih = _BitmapInfoHeader(C.sizeof(_BitmapInfoHeader), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
        bits = C.c_void_p()
        bmp = _gdi.CreateDIBSection(dc, C.byref(bih), 0, C.byref(bits), None, 0)
        old_bmp = _gdi.SelectObject(dc, bmp)
        try:
            _gdi.SetTextColor(dc, 0xFFFFFF)
            _gdi.SetBkColor(dc, 0)
            _gdi.TextOutW(dc, pad, pad, text, len(text))
            _gdi.GdiFlush()
            buf = C.cast(bits, C.POINTER(C.c_uint8))
            # Green channel: identical to luma for white text, and free of ClearType's R/B split.
            return np.ctypeslib.as_array(buf, shape=(h, w, 4))[:, :, 1].copy()
        finally:
            _gdi.SelectObject(dc, old_bmp)
            _gdi.DeleteObject(bmp)
    finally:
        _gdi.SelectObject(dc, old_font)
        _gdi.DeleteObject(font)
        _gdi.DeleteDC(dc)


FR_PRIVATE = 0x10


def load_private_font(path: Path) -> bool:
    """Make a font file renderable by this process only (nothing is installed)."""
    return _gdi.AddFontResourceExW(str(path), FR_PRIVATE, None) > 0


class _LogFontW(C.Structure):
    _fields_ = [
        ("lfHeight", W.LONG),
        ("lfWidth", W.LONG),
        ("lfEscapement", W.LONG),
        ("lfOrientation", W.LONG),
        ("lfWeight", W.LONG),
        ("lfItalic", W.BYTE),
        ("lfUnderline", W.BYTE),
        ("lfStrikeOut", W.BYTE),
        ("lfCharSet", W.BYTE),
        ("lfOutPrecision", W.BYTE),
        ("lfClipPrecision", W.BYTE),
        ("lfQuality", W.BYTE),
        ("lfPitchAndFamily", W.BYTE),
        ("lfFaceName", W.WCHAR * 32),
    ]


_ENUMPROC = C.WINFUNCTYPE(C.c_int, C.POINTER(_LogFontW), C.c_void_p, W.DWORD, W.LPARAM)
_gdi.EnumFontFamiliesExW.argtypes = [W.HDC, C.POINTER(_LogFontW), _ENUMPROC, W.LPARAM, W.DWORD]


def font_families() -> list[str]:
    """Installed (and privately loaded) font family names, excluding vertical '@' faces."""
    names: set[str] = set()

    def cb(lf: C._Pointer[_LogFontW], _tm: int, _type: int, _lp: int) -> int:
        name = lf.contents.lfFaceName
        if name and not name.startswith("@"):
            names.add(name)
        return 1

    dc = _gdi.CreateCompatibleDC(None)
    try:
        lf = _LogFontW()
        lf.lfCharSet = 1  # DEFAULT_CHARSET: enumerate every family once per charset
        _gdi.EnumFontFamiliesExW(dc, C.byref(lf), _ENUMPROC(cb), 0, 0)
    finally:
        _gdi.DeleteDC(dc)
    return sorted(names)
