//! Read the actual system-rendered cursor bitmap via undocumented SkyLight
//! symbols. ALL touches to private CoreGraphics/SkyLight symbols live in this
//! file — when Apple changes the API in a future macOS release (renames a
//! function, drops a parameter, returns a different type), update only this
//! file.
//!
//! Why this exists: `NSCursor.currentSystemCursor` returns the calling
//! *process's* NSCursor stack, not the cursor the WindowServer is actually
//! compositing on screen. macrdp's process never installs an I-beam or
//! crosshair, so NSCursor just keeps reporting the default arrow — even when
//! the user is typing in a Safari text field and the screen shows an I-beam,
//! or when `screencapture -i` has put a crosshair on screen. The
//! WindowServer's connection exposes the real composited cursor via the
//! private functions below.
//!
//! Symbols live in `/System/Library/PrivateFrameworks/SkyLight.framework`,
//! aren't exported in the framework's `.tbd` stub, and weren't kept under
//! the legacy `CGS*` names — they're `SLS*` on modern macOS (12+). We
//! resolve them at runtime via `dlsym(RTLD_DEFAULT, ...)` instead of
//! link-time: if Apple removes or renames any of them in a future macOS,
//! the lookup returns `None`, this module returns `None`, and `cursor/mod.rs`
//! falls back to the public NSCursor path without the binary failing to
//! launch.
//!
//! Reverse-engineered signatures cross-checked against:
//!   - setomits/PrivateHeadersForMacOS — `SLSCursor.h`
//!   - macmade/CursorView — `SkyLight.h`
//!   - sapuapu/MouseCape (CGS-era naming, same shape)
#![cfg(target_os = "macos")]

use core_graphics::geometry::{CGPoint, CGRect};
use libc::{dlsym, RTLD_DEFAULT};
use std::ffi::{c_int, c_void, CString};
use std::sync::OnceLock;

#[allow(non_camel_case_types)]
type CGSConnectionID = c_int;

type SLSMainConnectionIDFn = unsafe extern "C" fn() -> CGSConnectionID;
type SLSGetGlobalCursorDataSizeFn = unsafe extern "C" fn(CGSConnectionID, *mut c_int) -> c_int;
// Signature (9 out params after the connection):
//   data:             buffer to fill (caller-allocated, size from -DataSize)
//   data_size:        IN size of buffer / OUT bytes actually written
//   row_bytes:        OUT row stride
//   rect:             OUT cursor bounding rect (size = bitmap dimensions)
//   hot_spot:         OUT hotspot in pixels relative to the cursor rect
//   depth:            OUT bits per pixel (typically 32)
//   components:       OUT components per pixel (typically 4)
//   bits_per_component: OUT bits per component (typically 8)
// Pixel format in memory is little-endian BGRA premultiplied — i.e., the
// CG-native bitmap layout (kCGBitmapByteOrder32Little |
// kCGImageAlphaPremultipliedFirst). Byte 0 of each pixel is blue.
type SLSGetGlobalCursorDataFn = unsafe extern "C" fn(
    CGSConnectionID,
    *mut c_void,
    *mut c_int,
    *mut c_int,
    *mut CGRect,
    *mut CGPoint,
    *mut c_int,
    *mut c_int,
    *mut c_int,
) -> c_int;

/// Resolve a private SkyLight/CoreGraphics symbol by name. Returns `None`
/// if the symbol isn't present (the OS version removed it, sandbox is
/// hiding it, etc.).
fn resolve<T: Copy>(name: &str) -> Option<T> {
    let cname = CString::new(name).ok()?;
    let sym = unsafe { dlsym(RTLD_DEFAULT, cname.as_ptr()) };
    if sym.is_null() {
        return None;
    }
    // SAFETY: caller picks the right function-pointer type for the
    // symbol they asked for. We're internal-only here.
    Some(unsafe { *(&sym as *const *mut c_void as *const T) })
}

fn main_connection_fn() -> Option<SLSMainConnectionIDFn> {
    static F: OnceLock<Option<SLSMainConnectionIDFn>> = OnceLock::new();
    *F.get_or_init(|| resolve("SLSMainConnectionID"))
}

fn cursor_data_size_fn() -> Option<SLSGetGlobalCursorDataSizeFn> {
    static F: OnceLock<Option<SLSGetGlobalCursorDataSizeFn>> = OnceLock::new();
    *F.get_or_init(|| resolve("SLSGetGlobalCursorDataSize"))
}

fn cursor_data_fn() -> Option<SLSGetGlobalCursorDataFn> {
    static F: OnceLock<Option<SLSGetGlobalCursorDataFn>> = OnceLock::new();
    *F.get_or_init(|| resolve("SLSGetGlobalCursorData"))
}

/// Snapshot the real WindowServer cursor.
///
/// Returns `(bitmap, width, height, hot_x, hot_y)`. The bitmap is tightly
/// packed, bottom row first, B,G,R,A with straight alpha: the RDP 32 bpp
/// pointer XOR mask that `RGBAPointer` carries as is (see
/// [`to_pointer_bitmap`]). Returns `None` if any private
/// symbol is missing or any SLS call fails.
pub fn copy_current_system_cursor() -> Option<(Vec<u8>, u16, u16, u16, u16)> {
    let main_conn = main_connection_fn()?;
    let size_fn = cursor_data_size_fn()?;
    let data_fn = cursor_data_fn()?;

    unsafe {
        let cid = main_conn();

        let mut data_size: c_int = 0;
        if size_fn(cid, &mut data_size) != 0 || data_size <= 0 {
            return None;
        }
        // Refuse absurd sizes — RDP pointers cap at 256x256x4 = 256 KiB
        // plus a generous margin. Anything wildly bigger is a wire-format
        // mismatch we'd rather skip than honour.
        if data_size as usize > 1024 * 1024 {
            return None;
        }

        let mut buf = vec![0u8; data_size as usize];
        let mut size_inout: c_int = data_size;
        let mut row_bytes: c_int = 0;
        let mut rect = CGRect {
            origin: CGPoint { x: 0.0, y: 0.0 },
            size: core_graphics::geometry::CGSize {
                width: 0.0,
                height: 0.0,
            },
        };
        let mut hot = CGPoint { x: 0.0, y: 0.0 };
        let mut depth: c_int = 0;
        let mut components: c_int = 0;
        let mut bits_per_component: c_int = 0;

        let err = data_fn(
            cid,
            buf.as_mut_ptr() as *mut c_void,
            &mut size_inout,
            &mut row_bytes,
            &mut rect,
            &mut hot,
            &mut depth,
            &mut components,
            &mut bits_per_component,
        );
        if err != 0 {
            return None;
        }

        let w = rect.size.width.round() as i64;
        let h = rect.size.height.round() as i64;
        if !(1..=MAX_SIDE as i64).contains(&w) || h < 1 {
            return None;
        }
        let w = w as usize;
        let h = h as usize;
        let row_bytes = row_bytes as usize;
        if row_bytes < w * 4 {
            return None;
        }
        // Sanity: BGRA premultiplied is what we expect (depth=32, comps=4,
        // bpc=8). If macOS ever returns something different — say a 16bpc
        // HDR cursor — skip rather than emit garbled colors.
        if depth != 32 || components != 4 || bits_per_component != 8 {
            return None;
        }
        if size_inout as usize > buf.len() {
            return None;
        }
        // An animated cursor comes as one tall strip of frames (see
        // `strip_frame_height`); send its first frame, as the NSCursor
        // image of the same cursor shows.
        let h = if h <= MAX_SIDE {
            h
        } else {
            strip_frame_height(&buf, row_bytes, w, h)?
        };

        let out = to_pointer_bitmap(&buf, row_bytes, w, h)?;

        let hot_x = hot.x.round().max(0.0).min(f64::from(u16::MAX)) as u16;
        let hot_y = hot.y.round().max(0.0).min(f64::from(u16::MAX)) as u16;
        Some((out, w as u16, h as u16, hot_x, hot_y))
    }
}

/// Convert the SkyLight cursor buffer to the RDP 32 bpp pointer bitmap.
///
/// SkyLight gives premultiplied little-endian BGRA (memory bytes B,G,R,A),
/// first row = top scan-line. The pointer XOR mask (MS-RDPBCGR 2.2.9.1.1.4.4)
/// wants straight-alpha B,G,R,A, first row = bottom scan-line. So keep the
/// byte order, remove the premultiplication and reverse the rows. `None` if
/// `buf` is too short.
fn to_pointer_bitmap(buf: &[u8], row_bytes: usize, w: usize, h: usize) -> Option<Vec<u8>> {
    let mut out = Vec::with_capacity(w * h * 4);
    for row in (0..h).rev() {
        let start = row * row_bytes;
        let src_row = buf.get(start..start + w * 4)?;
        for &[b, g, r, a] in src_row.as_chunks::<4>().0 {
            let (r, g, b) = un_premultiply(r, g, b, a);
            out.extend_from_slice(&[b, g, r, a]);
        }
    }
    Some(out)
}

/// Largest cursor side taken from SkyLight (the RDP pointer limit).
const MAX_SIDE: usize = 256;

/// Frame height of an animated cursor that SkyLight returns as one strip.
///
/// For an animated cursor (Chrome's `wait` / `progress`: an arrow with a
/// spinning disc), `SLSGetGlobalCursorData` gives all frames stacked in one
/// buffer: 28x600 on a 1x display = 15 frames of 28x40, the first frame in
/// the first rows. The call gives no frame count, so find it from the
/// pixels: the frames of one cursor cover the same pixels (alpha > 0) and
/// differ in colour and in the alpha of edge pixels. Return the smallest
/// divisor of `h` (at least 2 frames, at most [`MAX_SIDE`] rows) for which
/// each row covers the same pixels as the matching row of the first frame.
/// `None` when no such height exists, and when every row covers the same
/// pixels (a bar: no frame can be told apart, and a slice of it is not the
/// cursor).
fn strip_frame_height(buf: &[u8], row_bytes: usize, w: usize, h: usize) -> Option<usize> {
    if buf.len() < h * row_bytes {
        return None;
    }
    let covered = |row: usize, x: usize| buf[row * row_bytes + x * 4 + 3] != 0;
    let same_rows = |a: usize, b: usize| (0..w).all(|x| covered(a, x) == covered(b, x));
    if (1..h).all(|row| same_rows(row, 0)) {
        return None;
    }
    (2..=MAX_SIDE.min(h / 2))
        .filter(|&fh| h.is_multiple_of(fh))
        .find(|&fh| (fh..h).all(|row| same_rows(row, row % fh)))
}

fn un_premultiply(r: u8, g: u8, b: u8, a: u8) -> (u8, u8, u8) {
    match a {
        0 => (0, 0, 0),
        255 => (r, g, b),
        _ => {
            let a_u = a as u32;
            let r = ((r as u32 * 255 + a_u / 2) / a_u).min(255) as u8;
            let g = ((g as u32 * 255 + a_u / 2) / a_u).min(255) as u8;
            let b = ((b as u32 * 255 + a_u / 2) / a_u).min(255) as u8;
            (r, g, b)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A strip of `frames` frames of `w`x`fh` BGRA pixels. Each frame covers
    /// the same triangle; colour and edge alpha change per frame.
    fn strip(w: usize, fh: usize, frames: usize, row_bytes: usize) -> Vec<u8> {
        let mut buf = vec![0u8; row_bytes * fh * frames];
        for f in 0..frames {
            for y in 0..fh {
                for x in 0..w.min(y + 1) {
                    let p = (f * fh + y) * row_bytes + x * 4;
                    let edge = x == y.min(w - 1);
                    buf[p..p + 4].copy_from_slice(&[
                        f as u8 * 9,
                        0,
                        0,
                        if edge { 40 + f as u8 } else { 255 },
                    ]);
                }
            }
        }
        buf
    }

    #[test]
    fn pointer_bitmap_keeps_bgra_order_and_puts_the_bottom_row_first() {
        // 2x2, row_bytes 12 (padding 4). Top row: opaque red, half-alpha blue
        // (premultiplied 0x80 -> 0xff). Bottom row: transparent, opaque white.
        let buf = [
            0, 0, 255, 255, 128, 0, 0, 128, 9, 9, 9, 9, //
            0, 0, 0, 0, 255, 255, 255, 255, 9, 9, 9, 9,
        ];
        let out = to_pointer_bitmap(&buf, 12, 2, 2).unwrap();
        assert_eq!(
            out,
            [
                0, 0, 0, 0, 255, 255, 255, 255, // bottom row first
                0, 0, 255, 255, 255, 0, 0, 128, // red stays B=0,G=0,R=255
            ]
        );
        assert_eq!(to_pointer_bitmap(&buf[..19], 12, 2, 2), None);
    }

    #[test]
    fn strip_of_wait_cursor_frames_gives_the_frame_height() {
        // Chrome `cursor: wait` on a 1x display: 28x600, 15 frames of 28x40.
        let buf = strip(28, 40, 15, 112);
        assert_eq!(strip_frame_height(&buf, 112, 28, 600), Some(40));
        // Padded rows (row_bytes > w * 4) and a Retina-size strip.
        let buf = strip(56, 80, 15, 240);
        assert_eq!(strip_frame_height(&buf, 240, 56, 1200), Some(80));
    }

    #[test]
    fn tall_bar_with_the_same_rows_is_not_cut_into_frames() {
        // 28x300 solid bar, and the same bar with a colour gradient: every row
        // covers the same pixels, so no frame height (not even 1 or 2 rows).
        let mut buf = vec![0u8; 112 * 300];
        for (i, px) in buf.as_chunks_mut::<4>().0.iter_mut().enumerate() {
            *px = [0, 0, (i / 28) as u8, 255];
        }
        assert_eq!(strip_frame_height(&buf, 112, 28, 300), None);
        buf.as_chunks_mut::<4>()
            .0
            .iter_mut()
            .for_each(|px| px[2] = 9);
        assert_eq!(strip_frame_height(&buf, 112, 28, 300), None);
    }

    #[test]
    fn tall_image_without_repeated_frames_is_refused() {
        // One 28x300 shape (not a strip): no frame height fits.
        let buf = strip(28, 300, 1, 112);
        assert_eq!(strip_frame_height(&buf, 112, 28, 300), None);
        // Frames above the pointer limit: refused.
        let buf = strip(28, 300, 2, 112);
        assert_eq!(strip_frame_height(&buf, 112, 28, 600), None);
        // Short buffer: refused, no panic.
        assert_eq!(strip_frame_height(&buf[..100], 112, 28, 600), None);
    }
}
