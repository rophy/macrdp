// Performance benchmarks for macrdp video pipeline at 4K@120fps
//
// Target: 8.33ms per frame (1/120s) for 120fps feasibility.
// Run with: cargo bench -p macrdp-encode

use criterion::{black_box, criterion_group, criterion_main, Criterion};
use macrdp_encode::color_convert::VImageConverter;
use macrdp_encode::{align16, OpenH264Encoder, VideoEncoder};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Generate a gradient test pattern that simulates real screen content.
/// Uniform data compresses trivially and does not represent real workloads.
fn generate_test_pattern(width: u32, height: u32, stride: usize) -> Vec<u8> {
    let mut bgra = vec![0u8; stride * height as usize];
    for y in 0..height as usize {
        for x in 0..width as usize {
            let offset = y * stride + x * 4;
            bgra[offset] = (x % 256) as u8; // B
            bgra[offset + 1] = (y % 256) as u8; // G
            bgra[offset + 2] = ((x + y) % 256) as u8; // R
            bgra[offset + 3] = 255; // A
        }
    }
    bgra
}

/// Scalar reference BGRA->YUV420 for comparison against vImage SIMD path.
/// Intentionally simple — matches the kind of loop the old code used.
fn scalar_bgra_to_yuv420(bgra: &[u8], width: u32, height: u32, stride: usize, yuv: &mut [u8]) {
    let w = width as usize;
    let h = height as usize;
    let y_size = w * h;
    let uv_w = w / 2;

    // Y plane
    for row in 0..h {
        for col in 0..w {
            let px = row * stride + col * 4;
            let b = bgra[px] as i32;
            let g = bgra[px + 1] as i32;
            let r = bgra[px + 2] as i32;
            yuv[row * w + col] = ((77 * r + 150 * g + 29 * b) >> 8).clamp(0, 255) as u8;
        }
    }

    // U and V planes (subsampled 2x2)
    for row in (0..h).step_by(2) {
        for col in (0..w).step_by(2) {
            let px = row * stride + col * 4;
            let b = bgra[px] as i32;
            let g = bgra[px + 1] as i32;
            let r = bgra[px + 2] as i32;
            let u_idx = y_size + (row / 2) * uv_w + col / 2;
            let v_idx = u_idx + uv_w * (h / 2);
            yuv[u_idx] = (((-43 * r - 85 * g + 128 * b) >> 8) + 128).clamp(0, 255) as u8;
            yuv[v_idx] = (((128 * r - 107 * g - 21 * b) >> 8) + 128).clamp(0, 255) as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// Color conversion benchmarks
// ---------------------------------------------------------------------------

fn bench_color_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("color_conversion");

    // -- 4K resolution: 3840 x 2160 ------------------------------------------
    let width = 3840u32;
    let height = 2160u32;
    let stride = width as usize * 4;
    let bgra = generate_test_pattern(width, height, stride);

    let converter = VImageConverter::new().expect("VImageConverter::new failed");

    // vImage BGRA -> I420 (4K)
    let mut yuv_i420 = vec![0u8; (width * height * 3 / 2) as usize];
    group.bench_function("vimage_bgra_to_i420_4k", |b| {
        b.iter(|| {
            converter
                .bgra_to_i420(&bgra, width, height, stride, &mut yuv_i420)
                .unwrap();
        })
    });

    // vImage BGRA -> NV12 (4K)
    let mut y_buf = vec![0u8; (width * height) as usize];
    let mut uv_buf = vec![0u8; (width * height / 2) as usize];
    group.bench_function("vimage_bgra_to_nv12_4k", |b| {
        b.iter(|| {
            converter
                .bgra_to_nv12(&bgra, width, height, stride, &mut y_buf, &mut uv_buf)
                .unwrap();
        })
    });

    // Scalar reference BGRA -> YUV420 (4K)
    let mut yuv_scalar = vec![0u8; (width * height * 3 / 2) as usize];
    group.bench_function("scalar_bgra_to_yuv420_4k", |b| {
        b.iter(|| {
            scalar_bgra_to_yuv420(&bgra, width, height, stride, &mut yuv_scalar);
        })
    });

    // -- 1080p for comparison -------------------------------------------------
    let w1080 = 1920u32;
    let h1080 = 1080u32;
    let s1080 = w1080 as usize * 4;
    let bgra_1080 = generate_test_pattern(w1080, h1080, s1080);

    let mut yuv_1080 = vec![0u8; (w1080 * h1080 * 3 / 2) as usize];
    group.bench_function("vimage_bgra_to_i420_1080p", |b| {
        b.iter(|| {
            converter
                .bgra_to_i420(&bgra_1080, w1080, h1080, s1080, &mut yuv_1080)
                .unwrap();
        })
    });

    let mut yuv_scalar_1080 = vec![0u8; (w1080 * h1080 * 3 / 2) as usize];
    group.bench_function("scalar_bgra_to_yuv420_1080p", |b| {
        b.iter(|| {
            scalar_bgra_to_yuv420(&bgra_1080, w1080, h1080, s1080, &mut yuv_scalar_1080);
        })
    });

    group.finish();
}

// ---------------------------------------------------------------------------
// OpenH264 encode benchmarks
// ---------------------------------------------------------------------------

fn bench_openh264_encode(c: &mut Criterion) {
    let mut group = c.benchmark_group("openh264_encode");
    group.sample_size(20); // fewer samples — encode is slow at 4K

    // 4K encode
    {
        let width = 3840u32;
        let height = 2160u32;
        let stride = width as usize * 4;
        let bgra = generate_test_pattern(width, height, stride);
        let mut encoder =
            OpenH264Encoder::new(align16(width), align16(height), 120.0, 50_000_000, false)
                .expect("Failed to create 4K OpenH264 encoder");

        group.bench_function("openh264_4k_120fps", |b| {
            b.iter(|| {
                encoder.encode_bgra(&bgra, width, height, stride).unwrap();
            })
        });
    }

    // 1080p encode
    {
        let width = 1920u32;
        let height = 1080u32;
        let stride = width as usize * 4;
        let bgra = generate_test_pattern(width, height, stride);
        let mut encoder =
            OpenH264Encoder::new(align16(width), align16(height), 120.0, 30_000_000, false)
                .expect("Failed to create 1080p OpenH264 encoder");

        group.bench_function("openh264_1080p_120fps", |b| {
            b.iter(|| {
                encoder.encode_bgra(&bgra, width, height, stride).unwrap();
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Scalar NV12 helper (BT.601 full-range, 2x2 averaged UV, NV12 interleaved)
// Same algorithm as create_nv12_from_bgra_fast in videotoolbox.rs
// ---------------------------------------------------------------------------

fn scalar_bgra_to_nv12(
    bgra: &[u8],
    width: usize,
    height: usize,
    stride: usize,
    y_buf: &mut [u8],
    uv_buf: &mut [u8],
) {
    // Y plane
    for row in 0..height {
        for col in 0..width {
            let px = row * stride + col * 4;
            let b = bgra[px] as i32;
            let g = bgra[px + 1] as i32;
            let r = bgra[px + 2] as i32;
            y_buf[row * width + col] = ((77 * r + 150 * g + 29 * b) >> 8).clamp(0, 255) as u8;
        }
    }

    // UV plane: 2x2 averaged, NV12 interleaved (U then V)
    let uv_w = width / 2;
    for row in (0..height).step_by(2) {
        for col in (0..width).step_by(2) {
            let p00 = row * stride + col * 4;
            let p01 = p00 + 4;
            let p10 = (row + 1) * stride + col * 4;
            let p11 = p10 + 4;

            let r00 = bgra[p00 + 2] as i32;
            let g00 = bgra[p00 + 1] as i32;
            let b00 = bgra[p00] as i32;
            let r01 = bgra[p01 + 2] as i32;
            let g01 = bgra[p01 + 1] as i32;
            let b01 = bgra[p01] as i32;
            let r10 = bgra[p10 + 2] as i32;
            let g10 = bgra[p10 + 1] as i32;
            let b10 = bgra[p10] as i32;
            let r11 = bgra[p11 + 2] as i32;
            let g11 = bgra[p11 + 1] as i32;
            let b11 = bgra[p11] as i32;

            let rb = (r00 + r01 + r10 + r11) >> 2;
            let gb = (g00 + g01 + g10 + g11) >> 2;
            let bb = (b00 + b01 + b10 + b11) >> 2;

            let uv_idx = (row / 2) * uv_w * 2 + col;
            uv_buf[uv_idx] = (((-43 * rb - 85 * gb + 128 * bb) >> 8) + 128).clamp(0, 255) as u8;
            uv_buf[uv_idx + 1] = (((128 * rb - 107 * gb - 21 * bb) >> 8) + 128).clamp(0, 255) as u8;
        }
    }
}

// ---------------------------------------------------------------------------
// NV12 conversion benchmarks: vImage SIMD vs scalar
// ---------------------------------------------------------------------------

fn nv12_conversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("nv12_conversion");

    let (w, h) = (3840u32, 2160u32);
    let stride = w as usize * 4;
    let bgra = generate_test_pattern(w, h, stride);

    // vImage path
    if let Ok(converter) = VImageConverter::new() {
        let mut y_buf = vec![0u8; (w * h) as usize];
        let mut uv_buf = vec![0u8; (w * h / 2) as usize];
        group.bench_function("vimage_bgra_to_nv12_4k", |b| {
            b.iter(|| {
                converter
                    .bgra_to_nv12(black_box(&bgra), w, h, stride, &mut y_buf, &mut uv_buf)
                    .unwrap();
            })
        });
    }

    // Scalar reference
    {
        let mut y_buf = vec![0u8; (w * h) as usize];
        let mut uv_buf = vec![0u8; (w * h / 2) as usize];
        group.bench_function("scalar_bgra_to_nv12_4k", |b| {
            b.iter(|| {
                scalar_bgra_to_nv12(
                    black_box(&bgra),
                    w as usize,
                    h as usize,
                    stride,
                    &mut y_buf,
                    &mut uv_buf,
                );
            })
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// Option::take vs Vec::clone
// ---------------------------------------------------------------------------

fn take_vs_clone(c: &mut Criterion) {
    let mut group = c.benchmark_group("take_vs_clone");

    for size in [1_000_000usize, 4_000_000] {
        let label = format!("{}MB", size / 1_000_000);
        let data = vec![42u8; size];

        group.bench_function(format!("vec_clone_{label}"), |b| {
            let source = data.clone();
            b.iter(|| {
                let _cloned = black_box(source.clone());
            })
        });

        group.bench_function(format!("option_take_{label}"), |b| {
            b.iter_batched(
                || Some(data.clone()),
                |mut opt| {
                    let _taken = black_box(opt.take());
                },
                criterion::BatchSize::SmallInput,
            )
        });
    }

    group.finish();
}

// ---------------------------------------------------------------------------
// VideoToolbox full path (macOS only)
// ---------------------------------------------------------------------------

#[cfg(target_os = "macos")]
fn vt_encode(c: &mut Criterion) {
    use macrdp_encode::{create_encoder, EncoderPreference, Quality};

    let mut group = c.benchmark_group("vt_encode");
    group.sample_size(20);

    for &(w, h, label) in &[(3840u32, 2160u32, "4k"), (1920, 1080, "1080p")] {
        let stride = w as usize * 4;
        let bgra = generate_test_pattern(w, h, stride);
        let bitrate = macrdp_encode::screen_bitrate(w, h, 30.0, Quality::Balanced);

        if let Ok(mut encoder) = create_encoder(
            w,
            h,
            30.0,
            Quality::Balanced,
            EncoderPreference::Hardware,
            false,
            bitrate,
        ) {
            group.bench_function(format!("vt_bgra_{label}_30fps"), |b| {
                b.iter(|| {
                    let _ = encoder.encode_bgra(black_box(&bgra), w, h, stride);
                })
            });
        }
    }

    group.finish();
}

// On non-macOS, provide an empty stub so criterion_group! compiles unconditionally.
#[cfg(not(target_os = "macos"))]
fn vt_encode(_c: &mut Criterion) {}

criterion_group!(
    benches,
    bench_color_conversion,
    bench_openh264_encode,
    nv12_conversion,
    take_vs_clone,
    vt_encode,
);
criterion_main!(benches);
