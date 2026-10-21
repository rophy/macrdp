//! AVC444 YUV plane split/combine helpers ([MS-RDPEGFX] §3.3.8.3.2, v1).
//!
//! The AVC444 server sends *two* H.264-encoded YUV420 frames per video frame: a
//! "main" view carrying full-resolution luma plus a 4:2:0 chroma subsample, and
//! an "auxiliary" view carrying the chroma samples that 4:2:0 subsampling
//! dropped. The receiver decodes both as YUV420, then merges them into a single
//! full-resolution YUV444 frame for display.
//!
//! - [`split_yuv444_to_yuv420_v1`] does the *encoder* side — full-res YUV444 →
//!   main + auxiliary YUV420 (planar; the reference of the tests).
//! - [`combine_luma_to_yuv444`] / [`combine_chroma_v1_to_yuv444`] do the
//!   *decoder* side (tests only). Both must be applied (luma first, then
//!   chroma) to fully reconstruct YUV444.
//! - [`Avc444Planes`] holds the YUV444 picture of one frame for the encoder
//!   (`videotoolbox.rs`), padded to whole macroblocks, and writes the two views
//!   into its NV12 buffers. The split above is its reference (tests).
//!
//! The combine functions are ported essentially verbatim from FreeRDP's
//! `general_LumaToYUV444` / `general_ChromaV1ToYUV444`
//! (`libfreerdp/primitives/prim_YUV.c`, Apache-2.0); they're the canonical
//! reference because every real AVC444 client in the wild today is FreeRDP- or
//! mstsc-based and decodes against the same scheme. The split is *not* a
//! verbatim port: FreeRDP's `general_YUV444SplitToYUV420` swaps U and V in the
//! main view's chroma (writes `pU[x] = pSrcV[2*x]`) which contradicts both the
//! §3.3.8.3.2 pseudo-code and FreeRDP's own combine, and is unreachable through
//! their normal RGB→AVC444YUV server path. This implementation follows the spec
//! pseudo-code instead, verified by the roundtrip test below.
//!
//! Buffer-size contract:
//! - **Width and height MUST be even.** [MS-RDPEGFX] §2.2.4.4 requires
//!   multiples of 16 on the H.264 wire (which implies even); the math here
//!   only needs even, but production callers should always pad to 16.
//! - Main view: Y = `width × height`, U/V = `(width / 2) × (height / 2)`.
//! - Auxiliary view: Y = `width × padded_aux_height(height)` (height is padded
//!   to a multiple of 16; the §3.3.8.3.2 B4/B5 packing uses 8-line strips and
//!   spec/FreeRDP both overpad), U/V = same as main.
//! - All planes are tightly packed with `stride = plane_width` in the test
//!   helpers; the production API takes explicit strides.

/// Height the auxiliary Y plane must be allocated to. Matches FreeRDP's
/// `padHeigth = height + 16 - height % 16`. Adds 16 when already aligned, which
/// is intentional padding margin — B4/B5 packing only writes through `height`
/// odd rows, so the over-allocation is just unused space.
#[cfg(test)]
pub fn padded_aux_height(height: usize) -> usize {
    height - height % 16 + 16
}

/// `v` rounded up to a multiple of 16 (whole H.264 macroblocks).
pub fn align16(v: usize) -> usize {
    v.div_ceil(16) * 16
}

/// One AVC444 frame on the encoder side: the full-range YUV444 picture at the
/// size rounded up to whole macroblocks, from which the main and auxiliary
/// views are written straight into the encoder's NV12 buffers
/// ([`Avc444Planes::fill_main`], [`Avc444Planes::fill_aux`]; the same layout as
/// [`split_yuv444_to_yuv420_v1`], without the planar copies). Allocated once
/// per encoder and reused for every frame.
///
/// The size is padded to multiples of 16 because the auxiliary view packs the
/// odd chroma rows in 16-row blocks (B4/B5): with a height that is a multiple
/// of 16 every packed row lands inside the picture; with another height the
/// last odd rows would fall outside the encoded frame.
pub struct Avc444Planes {
    width: usize,
    height: usize,
    y444: Vec<u8>,
    u444: Vec<u8>,
    v444: Vec<u8>,
}

impl Avc444Planes {
    /// Buffers for a `src_width` x `src_height` picture.
    pub fn new(src_width: usize, src_height: usize) -> Self {
        let width = align16(src_width.max(1));
        let height = align16(src_height.max(1));
        let full = width * height;
        Self {
            width,
            height,
            y444: vec![0; full],
            u444: vec![128; full],
            v444: vec![128; full],
        }
    }

    /// Encoded width (a multiple of 16).
    pub fn width(&self) -> usize {
        self.width
    }

    /// Encoded height (a multiple of 16).
    pub fn height(&self) -> usize {
        self.height
    }

    /// The YUV444 planes (stride = `width()`), to be filled by the converter.
    pub fn yuv444_mut(&mut self) -> (&mut [u8], &mut [u8], &mut [u8]) {
        (&mut self.y444, &mut self.u444, &mut self.v444)
    }

    /// Repeat the last column and row of the `src_width` x `src_height`
    /// picture into the padding, so the padded macroblocks cost few bits and
    /// do not bleed a wrong color into the picture's edge through the 4:2:0
    /// views.
    pub fn pad_edges(&mut self, src_width: usize, src_height: usize) {
        let (w, h) = (self.width, self.height);
        let sw = src_width.clamp(1, w);
        let sh = src_height.clamp(1, h);
        for plane in [&mut self.y444, &mut self.u444, &mut self.v444] {
            if sw < w {
                for row in plane.chunks_exact_mut(w).take(sh) {
                    let last = row[sw - 1];
                    row[sw..].fill(last);
                }
            }
            if sh < h {
                let (top, rest) = plane.split_at_mut(sh * w);
                let last = &top[(sh - 1) * w..];
                for row in rest.chunks_exact_mut(w) {
                    row.copy_from_slice(last);
                }
            }
        }
    }

    /// Write the main view into `y` (stride `y_stride`, `height()` rows) and
    /// `cbcr` (stride `c_stride`, `height() / 2` rows): B1 is Y; B2/B3 (NV12
    /// CbCr) is the rounded average of each 2x2 block of U and V, as a plain
    /// 4:2:0 picture has.
    ///
    /// Not the even-row, even-column sample: the clients rebuild that sample
    /// from the average. FreeRDP's 4:4:4 to RGB step
    /// (`general_YUV444ToRGB_DOUBLE_ROW`, `sse41_filter`) replaces it with
    /// `4 * main - (the 3 other samples of the block)` when that differs by 30
    /// or more, and FreeRDP's own encoder sends the average
    /// (`general_YUV444SplitToYUV420`, "use averaging"). A point sample
    /// there turned a colour edge wrong (one pixel of four per edge block).
    pub fn fill_main(&self, y: &mut [u8], y_stride: usize, cbcr: &mut [u8], c_stride: usize) {
        let w = self.width;
        for (row, src) in self.y444.chunks_exact(w).enumerate() {
            y[row * y_stride..row * y_stride + w].copy_from_slice(src);
        }
        for row in 0..self.height / 2 {
            let (top, bot) = (2 * row * w, (2 * row + 1) * w);
            let (u0, u1) = (&self.u444[top..top + w], &self.u444[bot..bot + w]);
            let (v0, v1) = (&self.v444[top..top + w], &self.v444[bot..bot + w]);
            let dst = &mut cbcr[row * c_stride..row * c_stride + w];
            let avg = |a: &[u8], b: &[u8], x: usize| {
                let sum =
                    u16::from(a[x]) + u16::from(a[x + 1]) + u16::from(b[x]) + u16::from(b[x + 1]);
                ((sum + 2) / 4) as u8
            };
            for (x, d) in dst.chunks_exact_mut(2).enumerate() {
                d[0] = avg(u0, u1, 2 * x);
                d[1] = avg(v0, v1, 2 * x);
            }
        }
    }

    /// Write the auxiliary view into `y` / `cbcr` (as [`Self::fill_main`]):
    /// B4/B5, the odd rows of U then V in 8-row strips of each 16-row block,
    /// as luma; B6/B7, U and V at even rows and odd columns, as chroma.
    pub fn fill_aux(&self, y: &mut [u8], y_stride: usize, cbcr: &mut [u8], c_stride: usize) {
        let (w, h) = (self.width, self.height);
        for row in 0..h {
            let (plane, k) = if row % 16 < 8 {
                (&self.u444, 8 * (row / 16) + row % 16)
            } else {
                (&self.v444, 8 * (row / 16) + row % 16 - 8)
            };
            let pos = 2 * k + 1;
            let dst = &mut y[row * y_stride..row * y_stride + w];
            if pos < h {
                dst.copy_from_slice(&plane[pos * w..(pos + 1) * w]);
            } else {
                dst.fill(128);
            }
        }
        // B6/B7: U and V at even rows and odd columns.
        for row in 0..h / 2 {
            let src = 2 * row * w;
            let u = &self.u444[src..src + w];
            let v = &self.v444[src..src + w];
            let dst = &mut cbcr[row * c_stride..row * c_stride + w];
            for (x, d) in dst.chunks_exact_mut(2).enumerate() {
                d[0] = u[2 * x + 1];
                d[1] = v[2 * x + 1];
            }
        }
    }
}

/// Split a full-resolution YUV444 frame into main + auxiliary YUV420 frames per
/// [MS-RDPEGFX] §3.3.8.3.2 v1. All slices are tightly packed with the given
/// strides; the caller is responsible for sizing them per the contract above.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn split_yuv444_to_yuv420_v1(
    src_y: &[u8],
    src_u: &[u8],
    src_v: &[u8],
    src_stride_y: usize,
    src_stride_u: usize,
    src_stride_v: usize,
    main_y: &mut [u8],
    main_u: &mut [u8],
    main_v: &mut [u8],
    main_stride_y: usize,
    main_stride_u: usize,
    main_stride_v: usize,
    aux_y: &mut [u8],
    aux_u: &mut [u8],
    aux_v: &mut [u8],
    aux_stride_y: usize,
    aux_stride_u: usize,
    aux_stride_v: usize,
    width: usize,
    height: usize,
) {
    debug_assert!(
        width.is_multiple_of(2) && height.is_multiple_of(2),
        "dimensions must be even"
    );
    let half_width = width / 2;
    let half_height = height / 2;
    let pad_height = padded_aux_height(height);

    // B1: main Y = src Y, full resolution.
    for y in 0..height {
        let src_row = &src_y[y * src_stride_y..y * src_stride_y + width];
        let dst_row = &mut main_y[y * main_stride_y..y * main_stride_y + width];
        dst_row.copy_from_slice(src_row);
    }

    // B2, B3: main U/V = even-row even-column samples (the standard 4:2:0
    // subsample). Per §3.3.8.3.2 pseudo-code: B2[x,y] = U444[2x, 2y].
    for y in 0..half_height {
        let src_u_row = &src_u[2 * y * src_stride_u..2 * y * src_stride_u + width];
        let src_v_row = &src_v[2 * y * src_stride_v..2 * y * src_stride_v + width];
        let dst_u_row = &mut main_u[y * main_stride_u..y * main_stride_u + half_width];
        let dst_v_row = &mut main_v[y * main_stride_v..y * main_stride_v + half_width];
        for x in 0..half_width {
            dst_u_row[x] = src_u_row[2 * x];
            dst_v_row[x] = src_v_row[2 * x];
        }
    }

    // B4, B5: odd-row samples of U then V, interleaved into the auxiliary Y
    // plane in 8-line strips. Within each 16-line block: first 8 lines carry
    // odd U rows, next 8 lines carry odd V rows. The pad_height overshoot is
    // skipped via `pos >= height` checks (matches FreeRDP behaviour).
    let mut u_y = 0usize;
    let mut v_y = 0usize;
    for y in 0..pad_height {
        if (y % 16) < 8 {
            let pos = 2 * u_y + 1;
            u_y += 1;
            if pos >= height {
                continue;
            }
            let src_row = &src_u[pos * src_stride_u..pos * src_stride_u + width];
            let dst_row = &mut aux_y[y * aux_stride_y..y * aux_stride_y + width];
            dst_row.copy_from_slice(src_row);
        } else {
            let pos = 2 * v_y + 1;
            v_y += 1;
            if pos >= height {
                continue;
            }
            let src_row = &src_v[pos * src_stride_v..pos * src_stride_v + width];
            let dst_row = &mut aux_y[y * aux_stride_y..y * aux_stride_y + width];
            dst_row.copy_from_slice(src_row);
        }
    }

    // B6, B7: even-row odd-column samples of U and V → auxiliary U and V planes
    // (themselves 4:2:0-shaped). Per §3.3.8.3.2: B6[x,y] = U444[2x+1, 2y].
    for y in 0..half_height {
        let src_u_row = &src_u[2 * y * src_stride_u..2 * y * src_stride_u + width];
        let src_v_row = &src_v[2 * y * src_stride_v..2 * y * src_stride_v + width];
        let dst_u_row = &mut aux_u[y * aux_stride_u..y * aux_stride_u + half_width];
        let dst_v_row = &mut aux_v[y * aux_stride_v..y * aux_stride_v + half_width];
        for x in 0..half_width {
            // 2x+1 is the odd column; clamp to width-1 on the right edge so a
            // non-even width doesn't read past the source row.
            let col = (2 * x + 1).min(width - 1);
            dst_u_row[x] = src_u_row[col];
            dst_v_row[x] = src_v_row[col];
        }
    }
}

/// Decoder LUMA pass — ported from FreeRDP's `general_LumaToYUV444`. Copies the
/// main-view Y to dst Y at full resolution and replicates the main-view U/V
/// samples across every position of each 2x2 block. The chroma pass below
/// overwrites the odd positions with the auxiliary samples.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn combine_luma_to_yuv444(
    main_y: &[u8],
    main_u: &[u8],
    main_v: &[u8],
    main_stride_y: usize,
    main_stride_u: usize,
    main_stride_v: usize,
    dst_y: &mut [u8],
    dst_u: &mut [u8],
    dst_v: &mut [u8],
    dst_stride_y: usize,
    dst_stride_u: usize,
    dst_stride_v: usize,
    width: usize,
    height: usize,
) {
    debug_assert!(
        width.is_multiple_of(2) && height.is_multiple_of(2),
        "dimensions must be even"
    );
    let half_width = width / 2;
    let half_height = height / 2;

    // B1: dst Y = main Y.
    for y in 0..height {
        let src_row = &main_y[y * main_stride_y..y * main_stride_y + width];
        let dst_row = &mut dst_y[y * dst_stride_y..y * dst_stride_y + width];
        dst_row.copy_from_slice(src_row);
    }

    // B2, B3: replicate main U/V into the four positions of each 2x2 block.
    for y in 0..half_height {
        let src_u_row = &main_u[y * main_stride_u..y * main_stride_u + half_width];
        let src_v_row = &main_v[y * main_stride_v..y * main_stride_v + half_width];
        let val_2y = 2 * y;
        let val_2y1 = val_2y + 1;
        // Slice both rows we'll write (guarded by even-height path; odd height
        // skips the second).
        let (dst_u_top, mut dst_u_bot) =
            split_two_rows_mut(dst_u, dst_stride_u, val_2y, val_2y1, width);
        let (dst_v_top, mut dst_v_bot) =
            split_two_rows_mut(dst_v, dst_stride_v, val_2y, val_2y1, width);
        for x in 0..half_width {
            let val_2x = 2 * x;
            let val_2x1 = val_2x + 1;
            let u = src_u_row[x];
            let v = src_v_row[x];
            dst_u_top[val_2x] = u;
            dst_v_top[val_2x] = v;
            if val_2x1 < width {
                dst_u_top[val_2x1] = u;
                dst_v_top[val_2x1] = v;
            }
            if let (Some(du), Some(dv)) = (dst_u_bot.as_deref_mut(), dst_v_bot.as_deref_mut()) {
                du[val_2x] = u;
                dv[val_2x] = v;
                if val_2x1 < width {
                    du[val_2x1] = u;
                    dv[val_2x1] = v;
                }
            }
        }
    }
}

/// Decoder CHROMAv1 pass — ported from FreeRDP's `general_ChromaV1ToYUV444`.
/// Reverses [`split_yuv444_to_yuv420_v1`]'s B4/B5/B6/B7 packing back into the
/// dst U/V planes' odd positions.
#[cfg(test)]
#[allow(clippy::too_many_arguments)]
pub fn combine_chroma_v1_to_yuv444(
    aux_y: &[u8],
    aux_u: &[u8],
    aux_v: &[u8],
    aux_stride_y: usize,
    aux_stride_u: usize,
    aux_stride_v: usize,
    dst_u: &mut [u8],
    dst_v: &mut [u8],
    dst_stride_u: usize,
    dst_stride_v: usize,
    width: usize,
    height: usize,
) {
    let half_width = width / 2;
    let half_height = height / 2;
    let pad_height = padded_aux_height(height);

    // B4, B5: auxiliary Y plane carries odd U rows in 8-line strips, then
    // odd V rows in the next 8 lines, per 16-line block.
    let mut u_y = 0usize;
    let mut v_y = 0usize;
    for y in 0..pad_height {
        if (y % 16) < 8 {
            let pos = 2 * u_y + 1;
            u_y += 1;
            if pos >= height {
                continue;
            }
            let src_row = &aux_y[y * aux_stride_y..y * aux_stride_y + width];
            let dst_row = &mut dst_u[pos * dst_stride_u..pos * dst_stride_u + width];
            dst_row.copy_from_slice(src_row);
        } else {
            let pos = 2 * v_y + 1;
            v_y += 1;
            if pos >= height {
                continue;
            }
            let src_row = &aux_y[y * aux_stride_y..y * aux_stride_y + width];
            let dst_row = &mut dst_v[pos * dst_stride_v..pos * dst_stride_v + width];
            dst_row.copy_from_slice(src_row);
        }
    }

    // B6, B7: even-row odd-column samples. Writes 2x+1 of each even dst row.
    for y in 0..half_height {
        let src_u_row = &aux_u[y * aux_stride_u..y * aux_stride_u + half_width];
        let src_v_row = &aux_v[y * aux_stride_v..y * aux_stride_v + half_width];
        let val_2y = 2 * y;
        let dst_u_row = &mut dst_u[val_2y * dst_stride_u..val_2y * dst_stride_u + width];
        let dst_v_row = &mut dst_v[val_2y * dst_stride_v..val_2y * dst_stride_v + width];
        for x in 0..half_width {
            let val_2x1 = 2 * x + 1;
            dst_u_row[val_2x1] = src_u_row[x];
            dst_v_row[val_2x1] = src_v_row[x];
        }
    }
}

/// Borrow two disjoint rows from a strided plane. The bottom row is `None`
/// when it falls outside `height` (caller's responsibility to verify).
#[cfg(test)]
fn split_two_rows_mut(
    plane: &mut [u8],
    stride: usize,
    top: usize,
    bot: usize,
    row_len: usize,
) -> (&mut [u8], Option<&mut [u8]>) {
    let top_off = top * stride;
    let bot_off = bot * stride;
    if bot_off + row_len > plane.len() {
        return (&mut plane[top_off..top_off + row_len], None);
    }
    let (lo, hi) = plane.split_at_mut(bot_off);
    (
        &mut lo[top_off..top_off + row_len],
        Some(&mut hi[..row_len]),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Generate a deterministic YUV444 source using a per-plane seeded LCG so
    /// each pixel has a distinct value with no correlation to its neighbours.
    /// Width/height are arbitrary (any positive values, even or odd).
    fn synthetic_yuv444(width: usize, height: usize) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let mut y = vec![0u8; width * height];
        let mut u = vec![0u8; width * height];
        let mut v = vec![0u8; width * height];
        let lcg = |seed: u64, n: usize| -> u8 {
            let mut s = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(n as u64);
            s ^= s >> 33;
            (s & 0xff) as u8
        };
        for j in 0..height {
            for i in 0..width {
                let idx = j * width + i;
                y[idx] = lcg(0x1111_1111, idx);
                u[idx] = lcg(0x2222_2222, idx);
                v[idx] = lcg(0x3333_3333, idx);
            }
        }
        (y, u, v)
    }

    /// End-to-end roundtrip: split full-res YUV444 → main + aux YUV420 →
    /// combine back to YUV444 → assert pixel-equality with the source.
    fn roundtrip(width: usize, height: usize) {
        let (src_y, src_u, src_v) = synthetic_yuv444(width, height);

        // Even dimensions enforced by the helper itself; matches the prod
        // contract (always padded to a multiple of 16).
        let half_w = width / 2;
        let half_h = height / 2;
        let pad_h = padded_aux_height(height);

        let mut main_y = vec![0u8; width * height];
        let mut main_u = vec![0u8; half_w * half_h];
        let mut main_v = vec![0u8; half_w * half_h];
        let mut aux_y = vec![0u8; width * pad_h];
        let mut aux_u = vec![0u8; half_w * half_h];
        let mut aux_v = vec![0u8; half_w * half_h];

        split_yuv444_to_yuv420_v1(
            &src_y,
            &src_u,
            &src_v,
            width,
            width,
            width,
            &mut main_y,
            &mut main_u,
            &mut main_v,
            width,
            half_w,
            half_w,
            &mut aux_y,
            &mut aux_u,
            &mut aux_v,
            width,
            half_w,
            half_w,
            width,
            height,
        );

        let mut dst_y = vec![0u8; width * height];
        let mut dst_u = vec![0u8; width * height];
        let mut dst_v = vec![0u8; width * height];

        combine_luma_to_yuv444(
            &main_y, &main_u, &main_v, width, half_w, half_w, &mut dst_y, &mut dst_u, &mut dst_v,
            width, width, width, width, height,
        );
        combine_chroma_v1_to_yuv444(
            &aux_y, &aux_u, &aux_v, width, half_w, half_w, &mut dst_u, &mut dst_v, width, width,
            width, height,
        );

        // Y is sent at full resolution; every byte must match.
        assert_eq!(dst_y, src_y, "Y plane mismatch ({width}x{height})");

        // U/V: every sample participated in the split — even-even via B2/B3,
        // even-odd via B6/B7, odd-* via B4/B5. The whole plane must roundtrip.
        for j in 0..height {
            for i in 0..width {
                let idx = j * width + i;
                assert_eq!(
                    dst_u[idx], src_u[idx],
                    "U mismatch at ({i},{j}) for {width}x{height}"
                );
                assert_eq!(
                    dst_v[idx], src_v[idx],
                    "V mismatch at ({i},{j}) for {width}x{height}"
                );
            }
        }
    }

    #[test]
    fn roundtrip_16x16() {
        roundtrip(16, 16);
    }

    #[test]
    fn roundtrip_64x48() {
        roundtrip(64, 48);
    }

    #[test]
    fn roundtrip_1920x1080() {
        roundtrip(1920, 1080);
    }

    /// Even, non-16-aligned sizes. AVC444's wire contract requires 16-alignment
    /// so production never sees these (we pad), but the split/combine math
    /// only needs even dimensions.
    #[test]
    fn roundtrip_even_non_16_aligned() {
        roundtrip(18, 16);
        roundtrip(34, 32);
        roundtrip(100, 64);
        roundtrip(64, 100);
    }

    /// The real Mac display size (1920x1080) needs height-padding to 1088 to
    /// be spec-legal for AVC444. Cover both: native 1920x1080 (even, just
    /// not 16-aligned in height) and the 16-aligned padded form.
    #[test]
    fn roundtrip_realistic_display_sizes() {
        roundtrip(1920, 1080);
        roundtrip(1920, 1088);
        roundtrip(2560, 1440);
        roundtrip(3840, 2160);
    }

    #[test]
    fn planes_are_whole_macroblocks() {
        let p = Avc444Planes::new(1920, 1080);
        assert_eq!((p.width(), p.height()), (1920, 1088));
        let p = Avc444Planes::new(1280, 720);
        assert_eq!((p.width(), p.height()), (1280, 720));
        let p = Avc444Planes::new(1001, 7);
        assert_eq!((p.width(), p.height()), (1008, 16));
        assert_eq!(align16(0), 0);
        assert_eq!(align16(16), 16);
        assert_eq!(align16(17), 32);
    }

    /// The padding repeats the last column and row; the picture is unchanged.
    #[test]
    fn pad_edges_repeats_the_last_column_and_row() {
        let mut p = Avc444Planes::new(5, 3);
        let w = p.width();
        {
            let (y, u, v) = p.yuv444_mut();
            for r in 0..3 {
                for c in 0..5 {
                    y[r * w + c] = (r * 10 + c) as u8;
                    u[r * w + c] = 100 + (r * 10 + c) as u8;
                    v[r * w + c] = 200 + (r * 10 + c) as u8;
                }
            }
        }
        p.pad_edges(5, 3);
        let (y, u, v) = p.yuv444_mut();
        for r in 0..16 {
            for c in 0..16 {
                let (sr, sc) = (r.min(2), c.min(4));
                let want = (sr * 10 + sc) as u8;
                assert_eq!(y[r * w + c], want, "Y ({c},{r})");
                assert_eq!(u[r * w + c], 100 + want, "U ({c},{r})");
                assert_eq!(v[r * w + c], 200 + want, "V ({c},{r})");
            }
        }
    }

    /// The views written into NV12 buffers equal the planar views of
    /// `split_yuv444_to_yuv420_v1` (the spec reference), and decode back to
    /// the source exactly.
    #[test]
    fn filled_views_match_the_reference_split() {
        let (w, h) = (64usize, 48usize);
        let mut p = Avc444Planes::new(w, h);
        let (src_y, src_u, src_v) = synthetic_yuv444(w, h);
        {
            let (y, u, v) = p.yuv444_mut();
            y.copy_from_slice(&src_y);
            u.copy_from_slice(&src_u);
            v.copy_from_slice(&src_v);
        }
        // NV12 with row padding, as CVPixelBuffer gives it.
        let (ys, cs) = (w + 8, w + 8);
        let mut my = vec![0u8; ys * h];
        let mut mc = vec![0u8; cs * h / 2];
        let mut ay = vec![0u8; ys * h];
        let mut ac = vec![0u8; cs * h / 2];
        p.fill_main(&mut my, ys, &mut mc, cs);
        p.fill_aux(&mut ay, ys, &mut ac, cs);
        let (hw, hh) = (w / 2, h / 2);
        let (mut ry, mut ru, mut rv) = (vec![0u8; w * h], vec![0u8; hw * hh], vec![0u8; hw * hh]);
        let mut ray = vec![0u8; w * padded_aux_height(h)];
        let (mut rau, mut rav) = (vec![0u8; hw * hh], vec![0u8; hw * hh]);
        split_yuv444_to_yuv420_v1(
            &src_y, &src_u, &src_v, w, w, w, &mut ry, &mut ru, &mut rv, w, hw, hw, &mut ray,
            &mut rau, &mut rav, w, hw, hw, w, h,
        );
        let planar = |y: &[u8], c: &[u8]| {
            let mut py = vec![0u8; w * h];
            let (mut pu, mut pv) = (vec![0u8; hw * hh], vec![0u8; hw * hh]);
            for r in 0..h {
                py[r * w..(r + 1) * w].copy_from_slice(&y[r * ys..r * ys + w]);
            }
            for r in 0..hh {
                for x in 0..hw {
                    pu[r * hw + x] = c[r * cs + 2 * x];
                    pv[r * hw + x] = c[r * cs + 2 * x + 1];
                }
            }
            (py, pu, pv)
        };
        let (py, pu, pv) = planar(&my, &mc);
        assert_eq!(py, ry);
        // Main-view chroma: the rounded 2x2 average (the reference split
        // point-samples it).
        let avg = |p: &[u8], r: usize, x: usize| {
            let s = u16::from(p[2 * r * w + 2 * x])
                + u16::from(p[2 * r * w + 2 * x + 1])
                + u16::from(p[(2 * r + 1) * w + 2 * x])
                + u16::from(p[(2 * r + 1) * w + 2 * x + 1]);
            ((s + 2) / 4) as u8
        };
        for r in 0..hh {
            for x in 0..hw {
                assert_eq!(pu[r * hw + x], avg(&src_u, r, x), "U ({x},{r})");
                assert_eq!(pv[r * hw + x], avg(&src_v, r, x), "V ({x},{r})");
            }
        }
        let (qy, qu, qv) = planar(&ay, &ac);
        assert_eq!(qy, ray[..w * h].to_vec());
        assert_eq!((qu, qv), (rau.clone(), rav.clone()));
        // Rows beyond the picture in the reference are untouched: nothing packed there.
        assert!(ray[w * h..].iter().all(|&b| b == 0));
        let mut dy = vec![0u8; w * h];
        let mut du = vec![0u8; w * h];
        let mut dv = vec![0u8; w * h];
        combine_luma_to_yuv444(
            &ry, &ru, &rv, w, hw, hw, &mut dy, &mut du, &mut dv, w, w, w, w, h,
        );
        combine_chroma_v1_to_yuv444(&ray, &rau, &rav, w, hw, hw, &mut du, &mut dv, w, w, w, h);
        assert_eq!((dy, du, dv), (src_y, src_u, src_v));
    }

    /// FreeRDP's 4:4:4 to RGB step (`general_YUV444ToRGB_DOUBLE_ROW`, same
    /// rule in its SSE path) rebuilds the even-row, even-column chroma sample
    /// from the main view as `4 * main - (other 3)` when that differs by 30
    /// or more. With the averaged main view a strong colour edge comes back
    /// within rounding and a weak one keeps the 4:2:0 error (below 30); a
    /// point sample is turned further off (blue on white: 0 instead of 90).
    fn freerdp_rebuild(main: u8, others: [u8; 3]) -> u8 {
        let sub: i32 = others.iter().map(|&o| i32::from(o)).sum();
        let out = (4 * i32::from(main) - sub).clamp(0, 255) as u8;
        if out.abs_diff(main) < 30 {
            main
        } else {
            out
        }
    }

    #[test]
    fn averaged_main_chroma_survives_the_freerdp_rebuild() {
        // Blue text on white, the edge in one 2x2 block: one sample 90, the
        // others 128 (and the reverse).
        // (sample, other three, error with the average, error with a point sample)
        for (u00, rest, avg_err, point_err) in [
            (90u8, 128u8, 29u8, 90u8),
            (200, 128, 0, 55),
            (128, 40, 0, 127),
            (0, 255, 0, 0),
        ] {
            let mut p = Avc444Planes::new(16, 16);
            {
                let (_, u, _) = p.yuv444_mut();
                u.fill(rest);
                u[0] = u00;
            }
            let (mut y, mut c) = (vec![0u8; 16 * 16], vec![0u8; 16 * 8]);
            p.fill_main(&mut y, 16, &mut c, 16);
            let got = freerdp_rebuild(c[0], [rest; 3]).abs_diff(u00);
            assert_eq!(got, avg_err, "u00 {u00} rest {rest}");
            assert_eq!(freerdp_rebuild(u00, [rest; 3]).abs_diff(u00), point_err);
        }
    }

    #[test]
    fn padded_aux_height_examples() {
        // Multiples of 16 still add 16 (FreeRDP defensive overpad).
        assert_eq!(padded_aux_height(16), 32);
        assert_eq!(padded_aux_height(64), 80);
        // Non-aligned rounds up to next multiple of 16.
        assert_eq!(padded_aux_height(60), 64);
        assert_eq!(padded_aux_height(1), 16);
        assert_eq!(padded_aux_height(17), 32);
    }
}
