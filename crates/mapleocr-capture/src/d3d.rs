//! D3D11 device setup and GPU-side ROI cropping.

use windows::Win32::Foundation::HMODULE;
use windows::Win32::Graphics::Direct3D::{D3D_DRIVER_TYPE_HARDWARE, D3D_DRIVER_TYPE_UNKNOWN, D3D_FEATURE_LEVEL_11_0};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BOX, D3D11_CPU_ACCESS_READ, D3D11_CREATE_DEVICE_BGRA_SUPPORT, D3D11_MAP_FLAG_DO_NOT_WAIT, D3D11_MAP_READ,
    D3D11_MAPPED_SUBRESOURCE, D3D11_SDK_VERSION, D3D11_TEXTURE2D_DESC, D3D11_USAGE_STAGING, D3D11CreateDevice,
    ID3D11Device, ID3D11DeviceContext, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::Common::{
    DXGI_FORMAT, DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_FORMAT_B8G8R8A8_UNORM_SRGB, DXGI_SAMPLE_DESC,
};
use windows::Win32::Graphics::Dxgi::{DXGI_ERROR_WAS_STILL_DRAWING, IDXGIAdapter};

use crate::{CaptureError, Rect, Result, RoiImage};

/// Create a hardware D3D11 device with BGRA support. Pass an adapter when the
/// device must live on a specific GPU (Desktop Duplication requires that).
pub fn create_device(adapter: Option<&IDXGIAdapter>) -> Result<(ID3D11Device, ID3D11DeviceContext)> {
    let mut device = None;
    let mut context = None;
    let driver = if adapter.is_some() { D3D_DRIVER_TYPE_UNKNOWN } else { D3D_DRIVER_TYPE_HARDWARE };
    // SAFETY: out-pointers are valid locals; on success both are Some.
    unsafe {
        D3D11CreateDevice(
            adapter,
            driver,
            HMODULE::default(),
            D3D11_CREATE_DEVICE_BGRA_SUPPORT,
            Some(&[D3D_FEATURE_LEVEL_11_0]),
            D3D11_SDK_VERSION,
            Some(&mut device),
            None,
            Some(&mut context),
        )?;
    }
    Ok((device.expect("device"), context.expect("context")))
}

struct Slot {
    tex: Option<ID3D11Texture2D>,
    /// Region actually copied this frame, in ROI-local pixels: (x, y, w, h).
    copied: (u32, u32, u32, u32),
}

/// Copies ROI rectangles out of a GPU texture into pre-allocated CPU buffers.
///
/// Each ROI gets its own small staging texture, so only ROI pixels cross the
/// bus. All copies are queued first and then mapped, so the GPU sees one
/// batch. `stage` does not allocate unless the source pixel format changes.
pub struct RoiStager {
    device: ID3D11Device,
    context: ID3D11DeviceContext,
    format: DXGI_FORMAT,
    slots: Vec<Slot>,
    images: Vec<RoiImage>,
    /// Inactive slots are neither copied nor mapped; their image keeps its last contents.
    active: Vec<bool>,
}

impl RoiStager {
    pub fn new(device: ID3D11Device, context: ID3D11DeviceContext) -> Self {
        Self {
            device,
            context,
            format: DXGI_FORMAT_B8G8R8A8_UNORM,
            slots: Vec::new(),
            images: Vec::new(),
            active: Vec::new(),
        }
    }

    pub fn set_rois(&mut self, rois: &[Rect]) -> Result<()> {
        self.images = rois.iter().map(|&r| RoiImage::new(r)).collect();
        self.slots = rois.iter().map(|_| Slot { tex: None, copied: (0, 0, 0, 0) }).collect();
        self.active = vec![true; rois.len()];
        self.create_textures()
    }

    /// Choose which ROIs the next `stage` copies. Shorter than the ROI list
    /// leaves the rest active.
    pub fn set_active(&mut self, active: &[bool]) {
        for (a, &b) in self.active.iter_mut().zip(active) {
            *a = b;
        }
    }

    fn create_textures(&mut self) -> Result<()> {
        for (slot, img) in self.slots.iter_mut().zip(&self.images) {
            slot.tex = None;
            if img.width() == 0 || img.height() == 0 {
                continue;
            }
            let desc = D3D11_TEXTURE2D_DESC {
                Width: img.width(),
                Height: img.height(),
                MipLevels: 1,
                ArraySize: 1,
                Format: self.format,
                SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
                Usage: D3D11_USAGE_STAGING,
                BindFlags: 0,
                CPUAccessFlags: D3D11_CPU_ACCESS_READ.0 as u32,
                MiscFlags: 0,
            };
            // SAFETY: desc is fully initialized; out-pointer is a valid Option.
            unsafe { self.device.CreateTexture2D(&desc, None, Some(&mut slot.tex))? };
        }
        Ok(())
    }

    pub fn images(&self) -> &[RoiImage] {
        &self.images
    }

    /// Crop every ROI out of `src`. `origin` is where client (0, 0) sits in
    /// `src`'s pixel space. ROI parts outside `src` come out zeroed.
    pub fn stage(&mut self, src: &ID3D11Texture2D, origin: (i32, i32)) -> Result<()> {
        let mut desc = D3D11_TEXTURE2D_DESC::default();
        // SAFETY: valid out-pointer.
        unsafe { src.GetDesc(&mut desc) };
        if desc.Format != self.format {
            if desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM && desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM_SRGB {
                return Err(CaptureError::Unsupported("source texture is not BGRA8"));
            }
            self.format = desc.Format;
            self.create_textures()?;
        }
        let (sw, sh) = (desc.Width as i64, desc.Height as i64);

        for ((slot, img), _) in self.slots.iter_mut().zip(&self.images).zip(&self.active).filter(|(_, a)| **a) {
            let r = img.rect();
            let x0 = origin.0 as i64 + r.x as i64;
            let y0 = origin.1 as i64 + r.y as i64;
            let cx0 = x0.clamp(0, sw);
            let cy0 = y0.clamp(0, sh);
            let cx1 = (x0 + r.w as i64).clamp(0, sw);
            let cy1 = (y0 + r.h as i64).clamp(0, sh);
            let (dx, dy) = ((cx0 - x0) as u32, (cy0 - y0) as u32);
            let (cw, ch) = ((cx1 - cx0).max(0) as u32, (cy1 - cy0).max(0) as u32);
            slot.copied = (dx, dy, cw, ch);
            let Some(tex) = &slot.tex else { continue };
            if cw == 0 || ch == 0 {
                continue;
            }
            let bx = D3D11_BOX {
                left: cx0 as u32,
                top: cy0 as u32,
                front: 0,
                right: cx1 as u32,
                bottom: cy1 as u32,
                back: 1,
            };
            // SAFETY: both textures share the format; the box is clamped to src
            // and the destination offset + size fits the ROI-sized staging texture.
            unsafe { self.context.CopySubresourceRegion(tex, 0, dx, dy, 0, src, 0, Some(&bx)) };
        }

        for ((slot, img), _) in self.slots.iter().zip(self.images.iter_mut()).zip(&self.active).filter(|(_, a)| **a) {
            let (dx, dy, cw, ch) = slot.copied;
            let full = cw == img.width() && ch == img.height();
            let row_bytes = img.width() as usize * 4;
            let out = img.data_mut();
            if !full {
                out.fill(0);
            }
            let Some(tex) = &slot.tex else { continue };
            if cw == 0 || ch == 0 {
                continue;
            }
            let mut mapped = D3D11_MAPPED_SUBRESOURCE::default();
            // Don't let the driver spin while the GPU (busy rendering the game)
            // finishes our copies: poll without waiting. Yield for the first
            // millisecond (copies usually land in ~0.4 ms), then sleep.
            // Measured: blocking Map doubled capture CPU; sleeping right away
            // raised p50 staging latency from 0.46 to 2.5 ms.
            let t0 = std::time::Instant::now();
            loop {
                // SAFETY: staging texture with CPU read access; unmapped below.
                match unsafe {
                    self.context.Map(tex, 0, D3D11_MAP_READ, D3D11_MAP_FLAG_DO_NOT_WAIT.0 as u32, Some(&mut mapped))
                } {
                    Ok(()) => break,
                    Err(e) if e.code() == DXGI_ERROR_WAS_STILL_DRAWING => {
                        if t0.elapsed() < std::time::Duration::from_millis(1) {
                            std::thread::yield_now();
                        } else {
                            std::thread::sleep(std::time::Duration::from_micros(200));
                        }
                    }
                    Err(e) => return Err(e.into()),
                }
            }
            let pitch = mapped.RowPitch as usize;
            let base = mapped.pData as *const u8;
            let (x_off, copy_bytes) = (dx as usize * 4, cw as usize * 4);
            for row in dy as usize..(dy + ch) as usize {
                // SAFETY: row < texture height and x_off + copy_bytes <= width * 4 <= pitch.
                let src_row = unsafe { std::slice::from_raw_parts(base.add(row * pitch + x_off), copy_bytes) };
                let o = row * row_bytes + x_off;
                out[o..o + copy_bytes].copy_from_slice(src_row);
            }
            // SAFETY: matches the Map above.
            unsafe { self.context.Unmap(tex, 0) };
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_DRIVER_TYPE_WARP;
    use windows::Win32::Graphics::Direct3D11::{D3D11_SUBRESOURCE_DATA, D3D11_USAGE_DEFAULT};

    /// Software device so the test runs on GPU-less CI machines.
    fn warp_device() -> (ID3D11Device, ID3D11DeviceContext) {
        let (mut device, mut context) = (None, None);
        unsafe {
            D3D11CreateDevice(
                None,
                D3D_DRIVER_TYPE_WARP,
                HMODULE::default(),
                D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                None,
                D3D11_SDK_VERSION,
                Some(&mut device),
                None,
                Some(&mut context),
            )
            .unwrap();
        }
        (device.unwrap(), context.unwrap())
    }

    /// Pixel (x, y) of the source encodes its own coordinates.
    fn px(x: u32, y: u32) -> [u8; 4] {
        [x as u8, y as u8, (x >> 8) as u8 | ((y >> 8) as u8) << 4, 0xff]
    }

    fn source(device: &ID3D11Device, w: u32, h: u32) -> ID3D11Texture2D {
        let data: Vec<u8> = (0..h).flat_map(|y| (0..w).flat_map(move |x| px(x, y))).collect();
        let desc = D3D11_TEXTURE2D_DESC {
            Width: w,
            Height: h,
            MipLevels: 1,
            ArraySize: 1,
            Format: DXGI_FORMAT_B8G8R8A8_UNORM,
            SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
            Usage: D3D11_USAGE_DEFAULT,
            BindFlags: 0,
            CPUAccessFlags: 0,
            MiscFlags: 0,
        };
        let init = D3D11_SUBRESOURCE_DATA { pSysMem: data.as_ptr().cast(), SysMemPitch: w * 4, SysMemSlicePitch: 0 };
        let mut tex = None;
        unsafe { device.CreateTexture2D(&desc, Some(&init), Some(&mut tex)) }.unwrap();
        tex.unwrap()
    }

    fn pixel(img: &RoiImage, x: u32, y: u32) -> [u8; 4] {
        let o = (y * img.width() + x) as usize * 4;
        img.data()[o..o + 4].try_into().unwrap()
    }

    #[test]
    fn crops_rois_with_origin_and_zeroes_out_of_bounds() {
        let (device, context) = warp_device();
        let src = source(&device, 300, 200);
        let mut stager = RoiStager::new(device, context);
        let rois = [
            Rect::new(10, 20, 30, 5),    // fully inside
            Rect::new(-5, -3, 10, 6),    // hangs off top-left of the client
            Rect::new(280, 180, 40, 40), // hangs off bottom-right of the texture
            Rect::new(1000, 1000, 4, 4), // entirely outside
        ];
        stager.set_rois(&rois).unwrap();
        let origin = (7, 4); // client (0, 0) sits at texture (7, 4), like a window border
        stager.stage(&src, origin).unwrap();
        let imgs = stager.images();

        for y in 0..5 {
            for x in 0..30 {
                assert_eq!(pixel(&imgs[0], x, y), px(10 + 7 + x, 20 + 4 + y));
            }
        }
        // Texture x = -5 + 7 + lx, y = -3 + 4 + ly; negative texture coords are zero.
        for ly in 0..6 {
            for lx in 0..10 {
                let (tx, ty) = (lx as i32 + 2, ly as i32 + 1);
                let want = if tx >= 0 && ty >= 0 { px(tx as u32, ty as u32) } else { [0; 4] };
                assert_eq!(pixel(&imgs[1], lx, ly), want, "roi1 ({lx},{ly})");
            }
        }
        // Texture x = 287 + lx (valid < 300), y = 184 + ly (valid < 200).
        for ly in 0..40 {
            for lx in 0..40 {
                let (tx, ty) = (287 + lx, 184 + ly);
                let want = if tx < 300 && ty < 200 { px(tx, ty) } else { [0; 4] };
                assert_eq!(pixel(&imgs[2], lx, ly), want, "roi2 ({lx},{ly})");
            }
        }
        assert!(imgs[3].data().iter().all(|&b| b == 0));
    }
}
