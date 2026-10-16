//! Shape of the system mouse cursor, for RDP pointer updates.

use std::ffi::c_void;

use core_graphics::base::{kCGBitmapByteOrder32Little, kCGImageAlphaPremultipliedFirst};
use core_graphics::color_space::CGColorSpace;
use core_graphics::context::{CGContext, CGInterpolationQuality};
use core_graphics::geometry::{CGPoint, CGRect, CGSize};
use core_graphics::image::CGImage;
use foreign_types::ForeignType;
use objc2::rc::autoreleasepool;
use objc2_app_kit::{NSBitmapImageRep, NSCursor};

/// Largest pointer side sent to the client (MS-RDPBCGR 2.2.7.2.7, 96x96 pointers).
pub const MAX_CURSOR_SIDE: u16 = 96;

/// A cursor image ready for an RDP 32 bpp pointer update.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CursorShape {
    pub width: u16,
    pub height: u16,
    pub hot_x: u16,
    pub hot_y: u16,
    /// BGRA, straight (not premultiplied) alpha, bottom row first.
    pub data: Vec<u8>,
}

impl CursorShape {
    /// No visible pixel (for example CSS `cursor: none` gives a transparent 1x1 image).
    pub fn is_invisible(&self) -> bool {
        self.data.as_chunks::<4>().0.iter().all(|p| p[3] == 0)
    }
}

/// Counter that WindowServer increments each time the system cursor changes.
/// It costs a few nanoseconds; read the image only when it changes.
pub fn cursor_seed() -> i32 {
    #[link(name = "CoreGraphics", kind = "framework")]
    extern "C" {
        fn CGSCurrentCursorSeed() -> i32;
    }
    // SAFETY: no arguments, returns a plain integer.
    unsafe { CGSCurrentCursorSeed() }
}

/// Read the system cursor and scale it by `scale` (RDP pixels per Mac point).
/// Returns `None` when macOS gives no cursor image.
pub fn current_cursor_shape(scale: f64) -> Option<CursorShape> {
    autoreleasepool(|_| {
        #[allow(deprecated)] // no public replacement gives the image of the system cursor
        let cursor = NSCursor::currentSystemCursor()?;
        let image = cursor.image();
        let size = image.size();
        let hot = cursor.hotSpot();
        let (width, height, scale) = target_size(size.width, size.height, scale)?;

        // The representation with the fewest pixels that is not smaller than the target.
        let reps = image.representations();
        let mut best: Option<(isize, objc2::rc::Retained<NSBitmapImageRep>)> = None;
        let mut largest: Option<(isize, objc2::rc::Retained<NSBitmapImageRep>)> = None;
        for rep in reps.iter() {
            let Ok(bitmap) = rep.downcast::<NSBitmapImageRep>() else {
                continue;
            };
            let w = bitmap.pixelsWide();
            if w >= width as isize && best.as_ref().is_none_or(|(bw, _)| w < *bw) {
                best = Some((w, bitmap.clone()));
            }
            if largest.as_ref().is_none_or(|(lw, _)| w > *lw) {
                largest = Some((w, bitmap));
            }
        }
        let (_, rep) = best.or(largest)?;
        let cg = rep.CGImage()?;
        let raw = objc2::rc::Retained::as_ptr(&cg) as *mut c_void;
        // SAFETY: `raw` is a valid CGImageRef; retain it for the core-graphics wrapper,
        // which releases it on drop.
        let image = unsafe {
            core_foundation_retain(raw);
            CGImage::from_ptr(raw.cast())
        };

        let (w, h) = (usize::from(width), usize::from(height));
        let mut ctx = CGContext::create_bitmap_context(
            None,
            w,
            h,
            8,
            w * 4,
            &CGColorSpace::create_device_rgb(),
            kCGImageAlphaPremultipliedFirst | kCGBitmapByteOrder32Little,
        );
        ctx.set_interpolation_quality(CGInterpolationQuality::CGInterpolationQualityHigh);
        ctx.draw_image(
            CGRect::new(&CGPoint::new(0.0, 0.0), &CGSize::new(w as f64, h as f64)),
            &image,
        );
        // Memory rows of the context are top first.
        let data = to_rdp_pixels(ctx.data(), w, h);
        Some(CursorShape {
            width,
            height,
            hot_x: scale_hotspot(hot.x, scale, width),
            hot_y: scale_hotspot(hot.y, scale, height),
            data,
        })
    })
}

fn core_foundation_retain(cf: *mut c_void) {
    #[link(name = "CoreFoundation", kind = "framework")]
    extern "C" {
        fn CFRetain(cf: *const c_void) -> *const c_void;
    }
    // SAFETY: `cf` is a valid CoreFoundation object.
    unsafe {
        CFRetain(cf);
    }
}

/// Pixel size of the scaled cursor, and the scale used (smaller than asked when
/// the cursor would be larger than [`MAX_CURSOR_SIDE`]).
fn target_size(w_pt: f64, h_pt: f64, scale: f64) -> Option<(u16, u16, f64)> {
    if !(w_pt > 0.0 && h_pt > 0.0 && scale > 0.0) {
        return None;
    }
    let max = f64::from(MAX_CURSOR_SIDE);
    let scale = scale.min(max / w_pt).min(max / h_pt);
    let w = (w_pt * scale).round().clamp(1.0, max) as u16;
    let h = (h_pt * scale).round().clamp(1.0, max) as u16;
    Some((w, h, scale))
}

fn scale_hotspot(v: f64, scale: f64, side: u16) -> u16 {
    ((v * scale).floor().max(0.0) as u16).min(side - 1)
}

/// Premultiplied BGRA, top row first -> straight-alpha BGRA, bottom row first.
fn to_rdp_pixels(src: &[u8], w: usize, h: usize) -> Vec<u8> {
    let stride = w * 4;
    let mut out = Vec::with_capacity(stride * h);
    for row in src[..stride * h].chunks_exact(stride).rev() {
        for p in row.as_chunks::<4>().0 {
            let a = p[3];
            if a == 0 {
                out.extend_from_slice(&[0, 0, 0, 0]);
            } else {
                let un =
                    |c: u8| ((u16::from(c) * 255 + u16::from(a) / 2) / u16::from(a)).min(255) as u8;
                out.extend_from_slice(&[un(p[0]), un(p[1]), un(p[2]), a]);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_size_scales_and_caps() {
        assert_eq!(target_size(17.0, 23.0, 1.0), Some((17, 23, 1.0)));
        assert_eq!(target_size(17.0, 23.0, 1.5), Some((26, 35, 1.5)));
        // 64 pt at 2x would be 128 px: limited to 96
        let (w, h, s) = target_size(64.0, 32.0, 2.0).unwrap();
        assert_eq!((w, h), (96, 48));
        assert!((s - 1.5).abs() < 1e-9);
        assert_eq!(target_size(0.0, 23.0, 1.0), None);
    }

    #[test]
    fn hotspot_scales_and_stays_inside() {
        assert_eq!(scale_hotspot(4.0, 1.5, 26), 6);
        assert_eq!(scale_hotspot(30.0, 1.0, 24), 23);
    }

    #[test]
    fn pixels_unpremultiply_and_flip() {
        // 1x2: top pixel half-transparent red (premultiplied), bottom transparent
        let src = [0, 0, 64, 128, 9, 9, 9, 0];
        let out = to_rdp_pixels(&src, 1, 2);
        assert_eq!(out, vec![0, 0, 0, 0, 0, 0, 128, 128]);
    }

    #[test]
    fn invisible_shape() {
        let s = CursorShape {
            width: 1,
            height: 1,
            hot_x: 0,
            hot_y: 0,
            data: vec![255, 255, 255, 0],
        };
        assert!(s.is_invisible());
    }
}
