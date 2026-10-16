//! Pointer shape updates for the client, with the RDP pointer cache.
//!
//! A new shape goes to the client as a New Pointer Update with a cache index;
//! a shape the client already has goes as a Cached Pointer Update (2 bytes).

use ironrdp_server::{DisplayUpdate, RGBAPointer};
use macrdp_capture::cursor::CursorShape;

/// Cache slots used on the client. The client gives its cache size in the
/// Pointer Capability Set (MS-RDPBCGR 2.2.7.1.5; mstsc gives 21), and the
/// display does not see the value, so stay below it.
pub const POINTER_CACHE_SLOTS: usize = 20;

/// Shapes in the client's pointer cache, and the shape the client shows.
#[derive(Default)]
pub struct PointerCache {
    /// (shape, last use) per cache index
    slots: Vec<(CursorShape, u64)>,
    /// Cache index of the shape the client shows
    shown: Option<usize>,
    uses: u64,
}

impl PointerCache {
    /// Forget everything (new connection or reactivation: the client cache is empty).
    pub fn clear(&mut self) {
        self.slots.clear();
        self.shown = None;
    }

    /// The update that makes the client show `shape`, or `None` when it shows it already.
    pub fn show(&mut self, mut shape: CursorShape) -> Option<DisplayUpdate> {
        if shape.is_invisible() {
            // A hidden cursor goes as a transparent shape: mstsc ignores the Null
            // Pointer Update (its cursor stays as it was). Windows treats a cursor
            // with alpha 0 in all pixels as a cursor without alpha (mstsc then
            // shows a black pixel), so one pixel gets alpha 1 (not visible).
            shape.data[3] = 1;
        }
        self.uses += 1;
        if let Some(i) = self.slots.iter().position(|(s, _)| *s == shape) {
            self.slots[i].1 = self.uses;
            if self.shown == Some(i) {
                return None;
            }
            self.shown = Some(i);
            return Some(DisplayUpdate::CachedPointer(i as u16));
        }
        let i = if self.slots.len() < POINTER_CACHE_SLOTS {
            self.slots.push((shape.clone(), self.uses));
            self.slots.len() - 1
        } else {
            // Least recently used slot
            let i = self
                .slots
                .iter()
                .enumerate()
                .min_by_key(|(_, (_, used))| *used)
                .map(|(i, _)| i)
                .expect("slots are full");
            self.slots[i] = (shape.clone(), self.uses);
            i
        };
        self.shown = Some(i);
        Some(DisplayUpdate::RGBAPointer(RGBAPointer {
            cache_index: i as u16,
            width: shape.width,
            height: shape.height,
            hot_x: shape.hot_x,
            hot_y: shape.hot_y,
            data: shape.data,
        }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(tag: u8) -> CursorShape {
        CursorShape {
            width: 1,
            height: 1,
            hot_x: 0,
            hot_y: 0,
            data: vec![tag, 0, 0, 255],
        }
    }

    fn kind(u: Option<DisplayUpdate>) -> String {
        match u {
            None => "none".into(),
            Some(DisplayUpdate::RGBAPointer(p)) => format!("new {}", p.cache_index),
            Some(DisplayUpdate::CachedPointer(i)) => format!("cached {i}"),
            Some(_) => "other".into(),
        }
    }

    #[test]
    fn new_then_cached() {
        let mut c = PointerCache::default();
        assert_eq!(kind(c.show(shape(1))), "new 0");
        assert_eq!(kind(c.show(shape(1))), "none");
        assert_eq!(kind(c.show(shape(2))), "new 1");
        assert_eq!(kind(c.show(shape(1))), "cached 0");
        assert_eq!(kind(c.show(shape(2))), "cached 1");
    }

    #[test]
    fn hidden_and_back() {
        let mut c = PointerCache::default();
        assert_eq!(kind(c.show(shape(1))), "new 0");
        let none = CursorShape {
            data: vec![0, 0, 0, 0],
            ..shape(0)
        };
        match c.show(none.clone()) {
            Some(DisplayUpdate::RGBAPointer(p)) => {
                assert_eq!(p.cache_index, 1);
                assert_eq!(p.data, vec![0, 0, 0, 1]);
            }
            _ => panic!("expected a new transparent pointer"),
        }
        assert_eq!(kind(c.show(none)), "none");
        assert_eq!(kind(c.show(shape(1))), "cached 0");
    }

    #[test]
    fn full_cache_replaces_least_recently_used() {
        let mut c = PointerCache::default();
        for t in 0..POINTER_CACHE_SLOTS as u8 {
            assert_eq!(kind(c.show(shape(t))), format!("new {t}"));
        }
        // Slot 0 used again: slot 1 is now the oldest
        assert_eq!(kind(c.show(shape(0))), "cached 0");
        assert_eq!(kind(c.show(shape(200))), "new 1");
        assert_eq!(kind(c.show(shape(1))), "new 2");
    }

    #[test]
    fn clear_forgets_shapes() {
        let mut c = PointerCache::default();
        c.show(shape(1));
        c.clear();
        assert_eq!(kind(c.show(shape(1))), "new 0");
    }
}
