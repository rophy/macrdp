//! The client's pointer cache (MS-RDPBCGR 2.2.9.1.1.4.4 / 2.2.9.1.1.4.6).
//!
//! A new shape goes to the client as a New Pointer Update with a cache index;
//! a shape the client already has goes as a Cached Pointer Update (2 bytes).
//! The number of slots comes from the client's Pointer Capability Set
//! (`pointerCacheSize`, see [`set_client_cache_size`]). The cache lives in
//! the display stream's `CursorState`: a new connection and a reactivation
//! (which empties the client's cache) each start a new stream, so a new cache.
//!
//! Ported from macrdp-v1 72d28d7, e74928b, c40705e (crates/macrdp-server/src/
//! pointer.rs `PointerCache` and its tests), relicensed by the author.
//! Changed for the fork: the shape is scaled and made visible before it gets
//! here (`cursor::macos`), and a client without a pointer cache keeps the
//! fork's old behaviour (each change as a New Pointer Update in slot 0).

use std::sync::atomic::{AtomicU32, Ordering};

use ironrdp_server::{DisplayUpdate, RGBAPointer};

/// Most cache slots used, whatever the client offers (mstsc 21, FreeRDP 25,
/// IronRDP 32): each slot costs a shape compare on a miss.
const MAX_SLOTS: u16 = 32;

/// Slots before the client's capabilities are known (every client takes 20:
/// MS-RDPBCGR 2.2.7.1.5 says a client should advertise at least that).
const DEFAULT_SLOTS: u16 = 20;

/// `pointerCacheSize` of the current client; `UNKNOWN` before the first.
static CLIENT_CACHE_SIZE: AtomicU32 = AtomicU32::new(UNKNOWN);
const UNKNOWN: u32 = u32::MAX;

/// Record the pointer cache size the connecting client advertised (Pointer
/// Capability Set). Called once per connection, before its display stream.
pub fn set_client_cache_size(size: u16) {
    CLIENT_CACHE_SIZE.store(u32::from(size), Ordering::Relaxed);
}

/// The client's pointer cache size (or the default before it is known).
pub fn client_cache_size() -> u16 {
    match CLIENT_CACHE_SIZE.load(Ordering::Relaxed) {
        UNKNOWN => DEFAULT_SLOTS,
        n => u16::try_from(n).unwrap_or(u16::MAX),
    }
}

/// A pointer shape as sent: B,G,R,A, bottom row first (the 32 bpp XOR mask).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Shape {
    pub width: u16,
    pub height: u16,
    pub hot_x: u16,
    pub hot_y: u16,
    pub data: Vec<u8>,
}

/// Shapes in the client's pointer cache, and the shape the client shows.
#[derive(Debug, Default)]
pub struct PointerCache {
    /// Client `pointerCacheSize`; 0 = no pointer cache.
    cache_size: u16,
    /// (shape as sent, last use) per cache index
    slots: Vec<(Shape, u64)>,
    /// Cache index of the shape the client shows
    shown: Option<usize>,
    uses: u64,
}

impl PointerCache {
    pub fn new(cache_size: u16) -> Self {
        Self {
            cache_size,
            ..Self::default()
        }
    }

    pub fn cache_size(&self) -> u16 {
        self.cache_size
    }

    fn slots(&self) -> usize {
        usize::from(self.cache_size.clamp(1, MAX_SLOTS))
    }

    /// The update that makes the client show `shape`, or `None` when it shows
    /// it already.
    pub fn show(&mut self, shape: Shape) -> Option<DisplayUpdate> {
        self.uses += 1;
        if self.cache_size == 0 {
            // No cache on the client: the shape goes again each time, in slot 0.
            if self.slots.first().is_some_and(|(s, _)| *s == shape) {
                return None;
            }
            self.slots = vec![(shape.clone(), self.uses)];
            self.shown = Some(0);
            return Some(new_pointer(0, shape));
        }
        if let Some(i) = self.slots.iter().position(|(s, _)| *s == shape) {
            self.slots[i].1 = self.uses;
            if self.shown == Some(i) {
                return None;
            }
            self.shown = Some(i);
            return Some(DisplayUpdate::CachedPointer(i as u16));
        }
        let i = if self.slots.len() < self.slots() {
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
        Some(new_pointer(i, shape))
    }
}

fn new_pointer(cache_index: usize, shape: Shape) -> DisplayUpdate {
    DisplayUpdate::RGBAPointer(RGBAPointer {
        cache_index: cache_index as u16,
        width: shape.width,
        height: shape.height,
        hot_x: shape.hot_x,
        hot_y: shape.hot_y,
        data: shape.data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(tag: u8) -> Shape {
        Shape {
            width: 2,
            height: 2,
            hot_x: 0,
            hot_y: 0,
            data: vec![tag; 16],
        }
    }

    fn kind(u: Option<DisplayUpdate>) -> String {
        match u {
            None => "none".to_owned(),
            Some(DisplayUpdate::RGBAPointer(p)) => format!("new {}", p.cache_index),
            Some(DisplayUpdate::CachedPointer(i)) => format!("cached {i}"),
            Some(_) => "other".to_owned(),
        }
    }

    #[test]
    fn a_shape_shown_before_goes_as_a_cached_pointer() {
        let mut c = PointerCache::new(25);
        assert_eq!(kind(c.show(shape(1))), "new 0");
        assert_eq!(kind(c.show(shape(1))), "none");
        assert_eq!(kind(c.show(shape(2))), "new 1");
        assert_eq!(kind(c.show(shape(1))), "cached 0");
        assert_eq!(kind(c.show(shape(2))), "cached 1");
        assert_eq!(kind(c.show(shape(2))), "none");
    }

    #[test]
    fn a_full_cache_replaces_the_least_recently_used_shape() {
        let mut c = PointerCache::new(21);
        for t in 0..21 {
            assert_eq!(kind(c.show(shape(t))), format!("new {t}"));
        }
        assert_eq!(kind(c.show(shape(0))), "cached 0");
        // Slot 1 is now the least recently used one.
        assert_eq!(kind(c.show(shape(100))), "new 1");
        assert_eq!(kind(c.show(shape(0))), "cached 0");
    }

    #[test]
    fn the_slots_follow_the_client_cache_size() {
        for (cache_size, slots) in [(5u16, 5u8), (25, 25), (1, 1), (500, MAX_SLOTS as u8)] {
            let mut c = PointerCache::new(cache_size);
            for t in 0..slots {
                c.show(shape(t));
            }
            // All slots used: the next new shape replaces slot 0 (the oldest).
            assert_eq!(kind(c.show(shape(250))), "new 0", "cache size {cache_size}");
        }
    }

    #[test]
    fn a_client_without_a_pointer_cache_gets_each_shape_again() {
        let mut c = PointerCache::new(0);
        assert_eq!(kind(c.show(shape(1))), "new 0");
        assert_eq!(kind(c.show(shape(1))), "none");
        assert_eq!(kind(c.show(shape(2))), "new 0");
        assert_eq!(kind(c.show(shape(1))), "new 0");
    }

    #[test]
    fn the_client_cache_size_is_the_default_until_a_client_says() {
        // One test owns the global: set, read, back to unknown.
        CLIENT_CACHE_SIZE.store(UNKNOWN, Ordering::Relaxed);
        assert_eq!(client_cache_size(), DEFAULT_SLOTS);
        set_client_cache_size(25);
        assert_eq!(client_cache_size(), 25);
        set_client_cache_size(0);
        assert_eq!(client_cache_size(), 0);
        CLIENT_CACHE_SIZE.store(UNKNOWN, Ordering::Relaxed);
    }
}
