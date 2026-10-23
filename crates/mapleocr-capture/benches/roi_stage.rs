//! GPU ROI crop cost: copy N small ROIs out of a 1080p BGRA texture and map
//! them into CPU buffers. This is the per-frame cost the capture backends add
//! on top of the OS handing us the frame.

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use mapleocr_capture::d3d::create_device;
use mapleocr_capture::{Rect, RoiStager};
use windows::Win32::Graphics::Direct3D11::{
    D3D11_BIND_SHADER_RESOURCE, D3D11_TEXTURE2D_DESC, D3D11_USAGE_DEFAULT, ID3D11Texture2D,
};
use windows::Win32::Graphics::Dxgi::Common::{DXGI_FORMAT_B8G8R8A8_UNORM, DXGI_SAMPLE_DESC};

fn rois(n: usize, w: u32, h: u32) -> Vec<Rect> {
    if w >= 1920 {
        return vec![Rect::new(0, 0, w, h)];
    }
    (0..n).map(|i| Rect::new(40 + (i as i32 % 8) * 220, 40 + (i as i32 / 8) * 60, w, h)).collect()
}

fn bench(c: &mut Criterion) {
    let (device, context) = create_device(None).expect("D3D11 device");
    let desc = D3D11_TEXTURE2D_DESC {
        Width: 1920,
        Height: 1080,
        MipLevels: 1,
        ArraySize: 1,
        Format: DXGI_FORMAT_B8G8R8A8_UNORM,
        SampleDesc: DXGI_SAMPLE_DESC { Count: 1, Quality: 0 },
        Usage: D3D11_USAGE_DEFAULT,
        BindFlags: D3D11_BIND_SHADER_RESOURCE.0 as u32,
        CPUAccessFlags: 0,
        MiscFlags: 0,
    };
    let mut src: Option<ID3D11Texture2D> = None;
    unsafe { device.CreateTexture2D(&desc, None, Some(&mut src)) }.expect("source texture");
    let src = src.unwrap();

    let mut g = c.benchmark_group("roi_stage_1080p");
    for &(n, w, h) in &[(1, 200, 24), (8, 200, 24), (32, 200, 24), (1, 1920, 1080)] {
        let mut stager = RoiStager::new(device.clone(), context.clone());
        stager.set_rois(&rois(n, w, h)).unwrap();
        g.bench_with_input(BenchmarkId::from_parameter(format!("{n}x{w}x{h}")), &n, |b, _| {
            b.iter(|| stager.stage(&src, (0, 0)).unwrap())
        });
    }
    g.finish();
}

criterion_group!(benches, bench);
criterion_main!(benches);
