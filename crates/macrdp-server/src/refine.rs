//! Lossless refinement of a quiet screen (mixed mode).
//!
//! H.264 AVC420 keeps only a quarter of the chroma, so colored text stays
//! blurred (MS-RDPEGFX clients show what the decoder gives). Windows hosts send
//! text with lossless codecs. Here each H.264 frame updates only the tiles that
//! changed (AVC420 region rectangles), and when the screen is quiet the changed
//! tiles that look like text or UI (few colors) are sent again with ClearCodec,
//! which is lossless. Later H.264 frames do not touch tiles that did not change,
//! so the lossless picture stays until that area changes again.

use macrdp_capture::Rect;

/// Nominal tile size. Tiles are made a little larger so that the screen splits
/// evenly (no thin edge tiles).
const TILE: u32 = 32;

/// A tile with more distinct colors than 1/`COLOR_SHARE` of its pixels is a
/// picture (photo, video) and stays H.264. Text and UI tiles have few colors
/// (anti-aliasing); photo tiles have almost one color per pixel. Measured on
/// 32x32 tiles of the bench page: text max 180, photo p10 843 of 1024.
pub const COLOR_SHARE: usize = 4;

/// ClearCodec uses its glyph cache for bitmaps of this many pixels or fewer;
/// lossless rectangles are kept larger than this.
pub const GLYPH_MAX_PIXELS: u32 = 1024;

/// A rectangle in pixels: x, y, width, height.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileRect {
    pub x: u16,
    pub y: u16,
    pub w: u16,
    pub h: u16,
}

/// A set of screen tiles.
#[derive(Debug, Clone)]
pub struct TileGrid {
    width: u32,
    height: u32,
    cols: u32,
    rows: u32,
    set: Vec<bool>,
}

impl TileGrid {
    pub fn new(width: u32, height: u32) -> Self {
        let cols = (width / TILE).max(1);
        let rows = (height / TILE).max(1);
        Self {
            width,
            height,
            cols,
            rows,
            set: vec![false; (cols * rows) as usize],
        }
    }

    pub fn size(&self) -> (u32, u32) {
        (self.width, self.height)
    }

    /// Left edge of column `c` (c == cols gives the width).
    fn x_edge(&self, c: u32) -> u32 {
        (u64::from(c) * u64::from(self.width) / u64::from(self.cols)) as u32
    }

    fn y_edge(&self, r: u32) -> u32 {
        (u64::from(r) * u64::from(self.height) / u64::from(self.rows)) as u32
    }

    /// Column that holds pixel column `x`.
    fn col_of(&self, x: u32) -> u32 {
        let mut c = (u64::from(x) * u64::from(self.cols) / u64::from(self.width.max(1))) as u32;
        c = c.min(self.cols - 1);
        // The edges are rounded down: step to the column that really holds x.
        while c + 1 < self.cols && self.x_edge(c + 1) <= x {
            c += 1;
        }
        while c > 0 && self.x_edge(c) > x {
            c -= 1;
        }
        c
    }

    fn row_of(&self, y: u32) -> u32 {
        let mut r = (u64::from(y) * u64::from(self.rows) / u64::from(self.height.max(1))) as u32;
        r = r.min(self.rows - 1);
        while r + 1 < self.rows && self.y_edge(r + 1) <= y {
            r += 1;
        }
        while r > 0 && self.y_edge(r) > y {
            r -= 1;
        }
        r
    }

    pub fn is_empty(&self) -> bool {
        !self.set.iter().any(|&b| b)
    }

    pub fn clear(&mut self) {
        self.set.fill(false);
    }

    pub fn mark_all(&mut self) {
        self.set.fill(true);
    }

    /// Mark the tiles that a frame's dirty rectangles touch. No rectangles
    /// means that the whole frame changed.
    pub fn mark_dirty(&mut self, rects: &[Rect]) {
        if rects.is_empty() {
            self.mark_all();
            return;
        }
        for r in rects {
            if r.width == 0 || r.height == 0 || r.x >= self.width || r.y >= self.height {
                continue;
            }
            // One pixel more on each side: the rectangles come from float
            // coordinates that were truncated.
            let x1 = (r.x.saturating_add(r.width)).min(self.width - 1);
            let y1 = (r.y.saturating_add(r.height)).min(self.height - 1);
            let (c0, c1) = (self.col_of(r.x.saturating_sub(1)), self.col_of(x1));
            let (r0, r1) = (self.row_of(r.y.saturating_sub(1)), self.row_of(y1));
            for row in r0..=r1 {
                for col in c0..=c1 {
                    self.set[(row * self.cols + col) as usize] = true;
                }
            }
        }
    }

    /// Add the tiles of another grid of the same size.
    pub fn union(&mut self, other: &TileGrid) {
        if other.set.len() != self.set.len() {
            self.mark_all();
            return;
        }
        for (a, b) in self.set.iter_mut().zip(&other.set) {
            *a |= *b;
        }
    }

    pub fn is_set(&self, col: u32, row: u32) -> bool {
        self.set[(row * self.cols + col) as usize]
    }

    pub fn unset(&mut self, col: u32, row: u32) {
        self.set[(row * self.cols + col) as usize] = false;
    }

    pub fn tile_rect(&self, col: u32, row: u32) -> TileRect {
        let (x0, x1) = (self.x_edge(col), self.x_edge(col + 1));
        let (y0, y1) = (self.y_edge(row), self.y_edge(row + 1));
        TileRect {
            x: x0 as u16,
            y: y0 as u16,
            w: (x1 - x0) as u16,
            h: (y1 - y0) as u16,
        }
    }

    /// The set tiles as rectangles: runs of set tiles in one tile row.
    pub fn rects(&self) -> Vec<TileRect> {
        self.runs(|col, row| self.set[(row * self.cols + col) as usize])
    }

    /// Like `runs`, but a run of `min_pixels` or fewer is made wider by the next
    /// tile (or the previous one at the right edge).
    pub fn runs_larger_than(
        &self,
        min_pixels: u32,
        pick: impl Fn(u32, u32) -> bool,
    ) -> Vec<TileRect> {
        let mut runs = self.runs(pick);
        for r in &mut runs {
            if u32::from(r.w) * u32::from(r.h) > min_pixels || self.cols == 1 {
                continue;
            }
            let row = self.row_of(u32::from(r.y));
            let first = self.col_of(u32::from(r.x));
            let last = self.col_of(u32::from(r.x + r.w) - 1);
            if last + 1 < self.cols {
                let next = self.tile_rect(last + 1, row);
                r.w = next.x + next.w - r.x;
            } else {
                let prev = self.tile_rect(first - 1, row);
                r.w += r.x - prev.x;
                r.x = prev.x;
            }
        }
        runs
    }

    /// Runs of tiles in one tile row for which `pick` is true.
    pub fn runs(&self, pick: impl Fn(u32, u32) -> bool) -> Vec<TileRect> {
        let mut out = Vec::new();
        for row in 0..self.rows {
            let mut col = 0;
            while col < self.cols {
                if !pick(col, row) {
                    col += 1;
                    continue;
                }
                let start = col;
                while col < self.cols && pick(col, row) {
                    col += 1;
                }
                let a = self.tile_rect(start, row);
                let b = self.tile_rect(col - 1, row);
                out.push(TileRect {
                    x: a.x,
                    y: a.y,
                    w: b.x + b.w - a.x,
                    h: a.h,
                });
            }
        }
        out
    }

    /// Set tiles as (col, row).
    pub fn tiles(&self) -> impl Iterator<Item = (u32, u32)> + '_ {
        let cols = self.cols;
        self.set
            .iter()
            .enumerate()
            .filter(|(_, &b)| b)
            .map(move |(i, _)| (i as u32 % cols, i as u32 / cols))
    }

    pub fn cols_rows(&self) -> (u32, u32) {
        (self.cols, self.rows)
    }
}

/// Whether a BGRA area looks like text or UI: at most 1/`COLOR_SHARE` of its
/// pixels have distinct colors (alpha ignored).
pub fn is_text_like(bgra: &[u8], stride: usize, r: TileRect) -> bool {
    few_colors(bgra, stride, r, (r.w as usize * r.h as usize) / COLOR_SHARE)
}

/// Whether a BGRA area has at most `max` distinct colors (alpha ignored).
pub fn few_colors(bgra: &[u8], stride: usize, r: TileRect, max: usize) -> bool {
    // Open-addressing set of 24-bit colors; u32::MAX marks a free slot.
    let size = (max * 2 + 2).next_power_of_two();
    let mask = size - 1;
    let mut table = vec![u32::MAX; size];
    let mut count = 0;
    for y in r.y as usize..(r.y + r.h) as usize {
        let row = &bgra[y * stride + r.x as usize * 4..y * stride + (r.x + r.w) as usize * 4];
        let mut prev = u32::MAX;
        for p in row.as_chunks::<4>().0 {
            let c = u32::from_le_bytes([p[0], p[1], p[2], 0]);
            if c == prev {
                continue;
            }
            prev = c;
            let mut i = (c.wrapping_mul(0x9E37_79B1) >> 8) as usize & mask;
            loop {
                if table[i] == c {
                    break;
                }
                if table[i] == u32::MAX {
                    table[i] = c;
                    count += 1;
                    if count > max {
                        return false;
                    }
                    break;
                }
                i = (i + 1) & mask;
            }
        }
    }
    true
}

/// Copy a rectangle out of a BGRA frame into a packed buffer.
pub fn copy_rect(bgra: &[u8], stride: usize, r: TileRect) -> Vec<u8> {
    let row_bytes = r.w as usize * 4;
    let mut out = Vec::with_capacity(row_bytes * r.h as usize);
    for y in r.y as usize..(r.y + r.h) as usize {
        let start = y * stride + r.x as usize * 4;
        out.extend_from_slice(&bgra[start..start + row_bytes]);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rect(x: u32, y: u32, width: u32, height: u32) -> Rect {
        Rect {
            x,
            y,
            width,
            height,
        }
    }

    #[test]
    fn tiles_cover_the_screen_evenly() {
        let g = TileGrid::new(1280, 720);
        assert_eq!(g.cols_rows(), (40, 22));
        let mut sum_w = 0;
        for c in 0..40 {
            sum_w += g.tile_rect(c, 0).w as u32;
        }
        let mut sum_h = 0;
        for r in 0..22 {
            let t = g.tile_rect(0, r);
            assert!(t.h >= 32, "tile height {}", t.h);
            sum_h += t.h as u32;
        }
        assert_eq!((sum_w, sum_h), (1280, 720));
        // Small screen: one tile.
        let g = TileGrid::new(50, 30);
        assert_eq!(g.cols_rows(), (1, 1));
        assert_eq!(
            g.tile_rect(0, 0),
            TileRect {
                x: 0,
                y: 0,
                w: 50,
                h: 30
            }
        );
    }

    #[test]
    fn dirty_rects_mark_touched_tiles() {
        let mut g = TileGrid::new(1280, 720);
        assert!(g.is_empty());
        g.mark_dirty(&[rect(70, 10, 10, 10)]);
        assert_eq!(g.tiles().collect::<Vec<_>>(), [(2, 0)]);
        // Across a tile edge: two tiles.
        g.clear();
        g.mark_dirty(&[rect(60, 0, 10, 1)]);
        assert_eq!(g.tiles().collect::<Vec<_>>(), [(1, 0), (2, 0)]);
        // One pixel of margin: a rectangle that ends at a tile edge touches the next tile.
        g.clear();
        g.mark_dirty(&[rect(65, 2, 31, 2)]);
        assert_eq!(g.tiles().collect::<Vec<_>>(), [(2, 0), (3, 0)]);
        // Last pixel of the screen; rectangles past the edge are clipped.
        g.clear();
        g.mark_dirty(&[rect(1279, 719, 50, 50), rect(5000, 0, 1, 1)]);
        assert_eq!(g.tiles().collect::<Vec<_>>(), [(39, 21)]);
        // No rectangles: the whole frame.
        g.clear();
        g.mark_dirty(&[]);
        assert_eq!(g.tiles().count(), 880);
    }

    #[test]
    fn rects_join_runs_in_a_row() {
        let mut g = TileGrid::new(1280, 720);
        g.mark_dirty(&[rect(0, 0, 95, 1), rect(641, 700, 1, 1)]);
        let r = g.rects();
        assert_eq!(
            r,
            [
                TileRect {
                    x: 0,
                    y: 0,
                    w: 96,
                    h: 32
                },
                TileRect {
                    x: 640,
                    y: 687,
                    w: 32,
                    h: 33
                },
            ]
        );
    }

    #[test]
    fn small_runs_get_wider() {
        let g = TileGrid::new(1280, 720);
        // Tile (5, 0) is 32x32 = 1024 pixels: one more tile to the right.
        let r = g.runs_larger_than(GLYPH_MAX_PIXELS, |c, r| (c, r) == (5, 0));
        assert_eq!(
            r,
            [TileRect {
                x: 160,
                y: 0,
                w: 64,
                h: 32
            }]
        );
        // At the right edge: one more tile to the left.
        let r = g.runs_larger_than(GLYPH_MAX_PIXELS, |c, r| (c, r) == (39, 0));
        assert_eq!(
            r,
            [TileRect {
                x: 1216,
                y: 0,
                w: 64,
                h: 32
            }]
        );
        // Larger runs stay.
        let r = g.runs_larger_than(GLYPH_MAX_PIXELS, |c, r| r == 0 && c < 2);
        assert_eq!(
            r,
            [TileRect {
                x: 0,
                y: 0,
                w: 64,
                h: 32
            }]
        );
    }

    #[test]
    fn color_count() {
        // 4x2 BGRA: 3 colors.
        let px = |b: u8| [b, 0, 0, 255];
        let data: Vec<u8> = [1, 1, 2, 3, 1, 2, 3, 3]
            .iter()
            .flat_map(|&b| px(b))
            .collect();
        let r = TileRect {
            x: 0,
            y: 0,
            w: 4,
            h: 2,
        };
        assert!(few_colors(&data, 16, r, 3));
        assert!(!few_colors(&data, 16, r, 2));
        // Alpha does not count.
        let mut d2 = data.clone();
        d2[3] = 0;
        assert!(few_colors(&d2, 16, r, 3));
        let c = copy_rect(
            &data,
            16,
            TileRect {
                x: 1,
                y: 1,
                w: 2,
                h: 1,
            },
        );
        assert_eq!(c, [px(2), px(3)].concat());
    }
}
