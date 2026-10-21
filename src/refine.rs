//! Lossless refinement of a quiet screen (mixed mode).
//!
//! Ported from macrdp-v1 (7f652c1, d8e3936, 349869d, 59be57b, b149b99,
//! c569680), relicensed by the author.
//!
//! H.264 AVC420 keeps only a quarter of the chroma, so colored text stays
//! blurred (MS-RDPEGFX clients show what the decoder gives). Windows hosts send
//! text with lossless codecs. Here each H.264 frame updates only the tiles that
//! changed (AVC420 region rectangles), and when the screen is quiet the changed
//! tiles that look like text or UI (few colors) are sent again with ClearCodec,
//! which is lossless. Later H.264 frames do not touch tiles that did not change,
//! so the lossless picture stays until that area changes again.
//!
//! This module has the pure parts (tile grid, tile diff, color test, chunking,
//! ClearCodec sequence numbers, ZGFX compression of a refine PDU) and the
//! per-connection state [`Refine`]. `crate::h264` wires it into the EGFX
//! pipeline.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use ironrdp_graphics::clearcodec::ClearCodecEncoder;
use ironrdp_graphics::zgfx;

/// Nominal tile size. Tiles are made a little larger so that the screen splits
/// evenly (no thin edge tiles).
const TILE: u32 = 32;

/// Lossless refinement starts when no H.264 frame went out for this long.
pub const REFINE_DELAY: Duration = Duration::from_millis(200);

/// Lossless refinement is sent in frames of about this size.
pub const REFINE_CHUNK_BYTES: usize = 64 * 1024;

/// How often to check again while the refinement waits (frames in flight).
pub const REFINE_POLL: Duration = Duration::from_millis(4);

/// A refinement frame that got no acknowledgement in this time is taken as
/// lost (or acknowledged in a way we did not see): the next one may go.
pub const REFINE_ACK_TIMEOUT: Duration = Duration::from_secs(1);

/// A tile with more distinct colors than 1/`COLOR_SHARE` of its pixels is a
/// picture (photo, video) and stays H.264. Text and UI tiles have few colors
/// (anti-aliasing); photo tiles have almost one color per pixel. Measured on
/// 32x32 tiles of the bench page: text max 180, photo p10 843 of 1024.
pub const COLOR_SHARE: usize = 4;

/// [`COLOR_SHARE`] for a scaled picture (client desktop larger or smaller
/// than the Mac display): the scaler blends the anti-aliased edges into more
/// colors. Measured on mstsc screenshots of the bench page: with 1/4, only
/// 50-60 % of the text tiles were sent lossless at 1.25x and 1.5x, and 52 % at
/// 0.8x. Simulation on the bench page (bilinear, bicubic, Lanczos; 32x32
/// tiles): 1.25x-1.5x, text max 445 colors of 1024, photo p10 662-811;
/// 0.6x-0.8x, text max 467, photo p10 800-919. A few photo tiles are below 512
/// and are sent lossless too (1.25x-1.5x: 2.6-4.6 %; 0.8x: 0.7-1.7 %; 0.6x-0.7x:
/// none). The cost is one ClearCodec tile each, once per quiet screen.
pub const SCALED_COLOR_SHARE: usize = 2;

/// The color share for a picture drawn at `pixel_scale` times the display
/// size in pixels (not in points: a Retina display has 2 pixels per point).
pub fn color_share(pixel_scale: f64) -> usize {
    if (pixel_scale - 1.0).abs() < 1e-3 {
        COLOR_SHARE
    } else {
        SCALED_COLOR_SHARE
    }
}

/// Scale of the captured picture against the display pixels: `session` is the
/// capture (session) size, `backing` the display size in pixels. With
/// `letterbox` the picture keeps the aspect ratio (the smaller factor);
/// stretched, any axis that is not 1:1 makes it a scaled picture.
pub fn capture_pixel_scale(session: (u16, u16), backing: (u16, u16), letterbox: bool) -> f64 {
    let sx = f64::from(session.0) / f64::from(backing.0.max(1));
    let sy = f64::from(session.1) / f64::from(backing.1.max(1));
    if letterbox {
        sx.min(sy)
    } else if (sx - 1.0).abs() >= 1e-3 {
        sx
    } else {
        sy
    }
}

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

    /// Whether every tile is set.
    pub fn is_full(&self) -> bool {
        self.set.iter().all(|&b| b)
    }

    pub fn clear(&mut self) {
        self.set.fill(false);
    }

    pub fn mark_all(&mut self) {
        self.set.fill(true);
    }

    /// Mark the tiles in which `cur` differs from `prev` (two BGRA frames of
    /// this grid's size with the same `stride`). Alpha counts too: the cost of
    /// a tile marked for nothing is one tile in the next H.264 region. Frames
    /// that are too short mark everything.
    pub fn mark_changed(&mut self, prev: &[u8], cur: &[u8], stride: usize) {
        let row_bytes = self.width as usize * 4;
        let usable = self.width > 0
            && self.height > 0
            && stride >= row_bytes
            && prev.len().min(cur.len()) >= stride * (self.height as usize - 1) + row_bytes;
        if !usable {
            self.mark_all();
            return;
        }
        let col_bytes: Vec<(usize, usize)> = (0..self.cols)
            .map(|c| (self.x_edge(c) as usize * 4, self.x_edge(c + 1) as usize * 4))
            .collect();
        for row in 0..self.rows {
            let (y0, y1) = (self.y_edge(row) as usize, self.y_edge(row + 1) as usize);
            let first = (row * self.cols) as usize;
            let tiles = &mut self.set[first..first + self.cols as usize];
            for y in y0..y1 {
                if tiles.iter().all(|&t| t) {
                    break;
                }
                let a = &prev[y * stride..y * stride + row_bytes];
                let b = &cur[y * stride..y * stride + row_bytes];
                if a == b {
                    continue;
                }
                for (c, &(x0, x1)) in col_bytes.iter().enumerate() {
                    if !tiles[c] && a[x0..x1] != b[x0..x1] {
                        tiles[c] = true;
                    }
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
    #[cfg(test)]
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

/// The tiles in which the captured frame `cur` (`width` x `height` pixels,
/// `stride` bytes per row) differs from `prev`, the frame captured before it.
/// `prev_ok` says that `prev` is a frame of the same size and stride; else
/// all tiles count as changed.
///
/// Both frames should be in normal memory: the capture loop copies the
/// capture buffer first (it keeps that copy anyway) and compares the copies.
/// Reading the capture buffer twice (compare, then copy) cost 2-3 % of a core
/// more in full-screen motion than one copy.
pub fn diff_frames(
    prev: &[u8],
    cur: &[u8],
    stride: usize,
    width: u32,
    height: u32,
    prev_ok: bool,
) -> TileGrid {
    let mut g = TileGrid::new(width, height);
    if prev_ok && prev.len() == cur.len() {
        g.mark_changed(prev, cur, stride);
    } else {
        g.mark_all();
    }
    g
}

/// Whether a BGRA area looks like text or UI: at most 1/`color_share` of its
/// pixels have distinct colors (alpha ignored). See [`COLOR_SHARE`].
pub fn is_text_like(bgra: &[u8], stride: usize, r: TileRect, color_share: usize) -> bool {
    few_colors(bgra, stride, r, (r.w as usize * r.h as usize) / color_share)
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

/// One chunk of lossless refinement (see `take_refine_chunk`).
pub struct RefineChunk {
    /// ClearCodec bitmaps (sequence numbers are set when they are sent)
    pub tiles: Vec<(TileRect, Vec<u8>)>,
    /// Pending tiles looked at
    pub total: usize,
    /// Of those, tiles sent lossless
    pub picked: usize,
}

/// Take whole tile rows from `pending` until the ClearCodec data reaches
/// `REFINE_CHUNK_BYTES`. Text-like tiles of those rows are encoded; the other
/// tiles (pictures) stay H.264 and leave `pending` too.
pub fn take_refine_chunk(
    pending: &mut TileGrid,
    data: &[u8],
    stride: usize,
    color_share: usize,
) -> RefineChunk {
    // A new encoder each time: its glyph cache starts empty, so it never
    // refers to a glyph the client may not have.
    let mut encoder = ClearCodecEncoder::new();
    let (cols, rows) = pending.cols_rows();
    let mut chunk = RefineChunk {
        tiles: Vec::new(),
        total: 0,
        picked: 0,
    };
    let mut bytes = 0;
    for row in 0..rows {
        let lossless: Vec<bool> = (0..cols)
            .map(|c| {
                pending.is_set(c, row)
                    && is_text_like(data, stride, pending.tile_rect(c, row), color_share)
            })
            .collect();
        for c in 0..cols {
            if pending.is_set(c, row) {
                pending.unset(c, row);
                chunk.total += 1;
                chunk.picked += usize::from(lossless[c as usize]);
            }
        }
        let runs =
            pending.runs_larger_than(GLYPH_MAX_PIXELS, |c, r| r == row && lossless[c as usize]);
        for rect in runs {
            let px = copy_rect(data, stride, rect);
            let encoded = encoder.encode(&px, rect.w, rect.h);
            bytes += encoded.len();
            chunk.tiles.push((rect, encoded));
        }
        if bytes >= REFINE_CHUNK_BYTES {
            break;
        }
    }
    chunk
}

/// Write sequence numbers from `first` into ClearCodec bitmap streams
/// (byte 0 is the flags, byte 1 the sequence number).
///
/// The client keeps one ClearCodec context for the connection, not one per
/// surface, and sets its counter back to 0 only on ResetGraphics (FreeRDP:
/// `clear_context_reset` from `gdi_ResetGraphics`). It refuses a bitmap with
/// another number (MS-RDPEGFX 2.2.4.1 `seqNumber`).
pub fn stamp_clear_seq(tiles: &mut [(TileRect, Vec<u8>)], first: u8) {
    for (i, (_, data)) in tiles.iter_mut().enumerate() {
        if let Some(seq) = data.get_mut(1) {
            *seq = first.wrapping_add(i as u8);
        }
    }
}

/// Largest uncompressed size of one ZGFX segment (MS-RDPEGFX 2.2.5.1).
const ZGFX_SEGMENT_MAX: usize = 65535;
/// RDP8 bulk compression type with the PACKET_COMPRESSED flag (2.2.5.2).
const ZGFX_FLAGS_COMPRESSED: u8 = 0x24;
/// RDP8 bulk compression type, raw data.
const ZGFX_FLAGS_RAW: u8 = 0x04;

/// ZGFX-compress one EGFX PDU that the pipeline wrapped uncompressed, with a
/// new compressor.
///
/// The pipeline sends PDUs uncompressed. A PDU compressed with a new
/// compressor refers only to its own bytes, so the client decodes it correctly
/// whatever else it put into its ZGFX history before, and in whatever order
/// the other messages reach it. A PDU larger than one segment goes as a
/// multipart message whose segments each decompress to at most 65535 bytes
/// (MS-RDPEGFX 3.1.9.1; FreeRDP refuses a larger segment), sharing the
/// compressor history in order. A segment that does not get smaller stays raw.
pub fn compress_wrapped(wrapped: &[u8]) -> Result<Vec<u8>, String> {
    // The wrapped PDU has raw segments only: a new decompressor unwraps it.
    let mut pdu = Vec::with_capacity(wrapped.len());
    zgfx::Decompressor::new()
        .decompress(wrapped, &mut pdu)
        .map_err(|e| format!("ZGFX unwrap: {e}"))?;
    let mut compressor = zgfx::Compressor::new();
    let mut segments: Vec<(bool, Vec<u8>)> = Vec::new();
    for chunk in pdu.chunks(ZGFX_SEGMENT_MAX) {
        let c = compressor
            .compress(chunk)
            .map_err(|e| format!("ZGFX compress: {e}"))?;
        if c.len() < chunk.len() {
            segments.push((true, c));
        } else {
            segments.push((false, chunk.to_vec()));
        }
    }
    let flags = |compressed: bool| {
        if compressed {
            ZGFX_FLAGS_COMPRESSED
        } else {
            ZGFX_FLAGS_RAW
        }
    };
    if let [(compressed, data)] = segments.as_slice() {
        // RDP_SEGMENTED_DATA, SEGMENTED_SINGLE
        let mut out = Vec::with_capacity(data.len() + 2);
        out.push(0xE0);
        out.push(flags(*compressed));
        out.extend_from_slice(data);
        return Ok(out);
    }
    // RDP_SEGMENTED_DATA, SEGMENTED_MULTIPART
    let count = u16::try_from(segments.len()).map_err(|_| "ZGFX: too many segments")?;
    let size = u32::try_from(pdu.len()).map_err(|_| "ZGFX: PDU too large")?;
    let mut out = Vec::with_capacity(7 + segments.iter().map(|(_, d)| d.len() + 5).sum::<usize>());
    out.push(0xE1);
    out.extend_from_slice(&count.to_le_bytes());
    out.extend_from_slice(&size.to_le_bytes());
    for (compressed, data) in &segments {
        let len = u32::try_from(data.len() + 1).map_err(|_| "ZGFX: segment too large")?;
        out.extend_from_slice(&len.to_le_bytes());
        out.push(flags(*compressed));
        out.extend_from_slice(data);
    }
    Ok(out)
}

/// What the next H.264 frame updates on the client (AVC420 regions).
#[derive(Debug, Clone)]
pub enum Regions {
    /// The whole frame
    Whole,
    /// These tiles
    Tiles(TileGrid),
}

/// Regions an encoded frame goes out with: `taken` (see [`Refine::take_regions`]),
/// but the whole frame for a keyframe. A keyframe is the loss backstop on a
/// lossy UDP tunnel (a lost frame leaves wrong tiles that are no longer dirty
/// or pending) and the TCP safety net for a client decode glitch, so it
/// repaints the whole surface as it did before the tile regions.
pub fn ship_regions(taken: Regions, keyframe: bool) -> Regions {
    if keyframe {
        Regions::Whole
    } else {
        taken
    }
}

/// Whether a refinement chunk can go now.
#[derive(Debug, PartialEq, Eq)]
pub enum RefineGate {
    Go,
    /// Frames are in flight: check again soon.
    Wait,
    /// Nothing to do until the next H.264 frame goes out (it sets a new time).
    Idle,
    /// The pipeline cannot take a chunk (no channel, no surface): drop the
    /// pending tiles.
    GiveUp,
}

/// Lossless refinement state of one EGFX connection.
///
/// The tiles go through three sets: `dirty` (changed since the last encoded
/// H.264 frame), the regions of encoded frames that wait for the ship thread
/// (`queued`, by presentation time stamp), and `pending` (shown from H.264 on
/// the client and not refined yet). An H.264 frame updates `dirty` and
/// `pending` (more is not needed, and the refined tiles must stay).
#[derive(Debug)]
pub struct Refine {
    size: (u32, u32),
    dirty: TileGrid,
    pending: TileGrid,
    queued: VecDeque<(i64, Regions)>,
    /// The last frame given to `submit_bgra` was encoded (or had nothing new):
    /// the client's H.264 picture is that frame.
    last_encoded: bool,
    /// When to refine (or check again)
    due: Option<Instant>,
    /// ClearCodec sequence number the client expects next
    seq: u8,
    /// Refinement frame sent and not acknowledged yet, and when it was sent
    chunk_in_flight: Option<(u32, Instant)>,
    /// Ids of the last refinement frames (their QoE reports are not about H.264)
    clear_ids: VecDeque<u32>,
    /// Counts surface resets (new surface, ResetGraphics, new encoder, live
    /// resize): a refinement frame taken before one must not go out after it.
    resets: u64,
}

/// What a refinement frame was taken for: the surface and the reset count
/// (see [`Refine::job_tag`]). It may go out only while both are current.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JobTag {
    pub surface_id: u16,
    pub resets: u64,
}

impl Default for Refine {
    fn default() -> Self {
        Self::new(0, 0)
    }
}

impl Refine {
    pub fn new(width: u32, height: u32) -> Self {
        let mut dirty = TileGrid::new(width, height);
        dirty.mark_all();
        Self {
            size: (width, height),
            dirty,
            pending: TileGrid::new(width, height),
            queued: VecDeque::new(),
            last_encoded: false,
            due: None,
            seq: 0,
            chunk_in_flight: None,
            clear_ids: VecDeque::new(),
            resets: 0,
        }
    }

    /// Use the grids of `width` x `height` (the surface size); a new size starts
    /// over with every tile dirty.
    fn fit(&mut self, width: u32, height: u32) {
        if self.size != (width, height) {
            let (seq, resets) = (self.seq, self.resets);
            *self = Self::new(width, height);
            self.seq = seq;
            self.resets = resets;
        }
    }

    /// A new surface is blank: the next H.264 frame covers all of it (also
    /// when that frame is not sent at the first try). Frames of the old surface
    /// that still wait are forgotten.
    pub fn on_new_surface(&mut self, width: u32, height: u32) {
        self.fit(width, height);
        self.resets += 1;
        self.dirty.mark_all();
        self.pending.clear();
        self.queued.clear();
        self.last_encoded = false;
        self.due = None;
        self.chunk_in_flight = None;
    }

    /// A new encoder starts its time stamps again: frames of the old one that
    /// still wait are forgotten, and the next frame repaints everything.
    pub fn on_new_encoder(&mut self) {
        self.resets += 1;
        self.queued.clear();
        self.dirty.mark_all();
    }

    /// ResetGraphics went out: the client's ClearCodec counter is 0 again.
    pub fn on_reset_graphics(&mut self) {
        self.resets += 1;
        self.seq = 0;
    }

    /// A captured frame changed the tiles `changed` (see [`diff_frames`]; None:
    /// all, a grid of another size: all): those tiles go into the next
    /// encoded frame.
    pub fn on_captured(&mut self, changed: Option<&TileGrid>) {
        self.last_encoded = false;
        match changed {
            Some(g) if g.size() == self.size => self.dirty.union(g),
            _ => self.dirty.mark_all(),
        }
    }

    /// Whether the tiles a new frame changed matter: while every tile is
    /// pending (full-screen motion), the next frame updates the whole surface
    /// anyway, so the capture loop need not compare the frames (it costs
    /// 2-3 % of a core in full-screen motion at 60 fps). A frame given without
    /// its changed tiles marks all of them, which is the same then.
    pub fn wants_changed_tiles(&self) -> bool {
        !self.pending.is_full()
    }

    /// Regions for the frame about to be encoded, or None when nothing on the
    /// client would change (the frame need not be sent). `whole` (a forced
    /// keyframe, a frame of another size) repaints everything.
    pub fn regions(&self, whole: bool) -> Option<Regions> {
        if whole || self.dirty.is_full() {
            return Some(Regions::Whole);
        }
        let mut grid = self.dirty.clone();
        grid.union(&self.pending);
        if grid.is_empty() {
            None
        } else if grid.is_full() {
            Some(Regions::Whole)
        } else {
            Some(Regions::Tiles(grid))
        }
    }

    /// The frame with time stamp `pts` was handed to the encoder with `regions`.
    pub fn on_encoded(&mut self, pts: i64, regions: Regions) {
        self.queued.push_back((pts, regions));
        self.dirty.clear();
        self.last_encoded = true;
    }

    /// The frame was not sent and had nothing new for the client.
    pub fn on_unchanged(&mut self) {
        self.last_encoded = true;
    }

    /// Regions of the encoded frame with time stamp `pts`. Frames the encoder
    /// did not deliver (older time stamps) add their regions to it. Unknown:
    /// the whole frame.
    pub fn take_regions(&mut self, pts: i64) -> Regions {
        let mut out: Option<Regions> = None;
        while self.queued.front().is_some_and(|(p, _)| *p <= pts) {
            let (_, r) = self.queued.pop_front().expect("checked");
            out = Some(match (out, r) {
                (None, r) => r,
                (Some(Regions::Tiles(mut a)), Regions::Tiles(b)) if a.size() == b.size() => {
                    a.union(&b);
                    Regions::Tiles(a)
                }
                _ => Regions::Whole,
            });
        }
        out.unwrap_or(Regions::Whole)
    }

    /// An H.264 frame with `regions` went out: those tiles now need
    /// refinement, after a quiet time.
    pub fn on_shipped(&mut self, regions: &Regions, now: Instant) {
        match regions {
            Regions::Whole => self.pending.mark_all(),
            Regions::Tiles(g) => self.pending.union(g),
        }
        self.due = Some(now + REFINE_DELAY);
    }

    /// When the refinement wants to run (None: nothing to refine).
    pub fn due(&self) -> Option<Instant> {
        if self.pending.is_empty() {
            None
        } else {
            self.due
        }
    }

    /// Check again in `REFINE_POLL`.
    pub fn poll_later(&mut self, now: Instant) {
        self.due = Some(now + REFINE_POLL);
    }

    /// The client no longer shows our tiles (closed channel, no surface).
    pub fn give_up(&mut self) {
        self.pending.clear();
        self.due = None;
        self.chunk_in_flight = None;
    }

    /// Whether a refinement chunk may be taken now, from the pipeline state:
    /// `ready` (channel ready, surface of the frame size), `pipeline_empty`
    /// (every encoded H.264 frame went out), `h264_in_flight` (H.264 frames
    /// sent and not acknowledged, 0 when the client does not acknowledge).
    pub fn gate(
        &mut self,
        ready: bool,
        pipeline_empty: bool,
        h264_in_flight: u32,
        now: Instant,
    ) -> RefineGate {
        if !ready {
            return RefineGate::GiveUp;
        }
        if let Some((_, sent)) = self.chunk_in_flight {
            if now.saturating_duration_since(sent) < REFINE_ACK_TIMEOUT {
                return RefineGate::Wait;
            }
            self.chunk_in_flight = None;
        }
        if !self.last_encoded || !pipeline_empty {
            // A frame is on its way (its shipping sets a new time), or the
            // last one was dropped (the next one sets it).
            self.due = None;
            return RefineGate::Idle;
        }
        if h264_in_flight > 0 {
            return RefineGate::Wait;
        }
        RefineGate::Go
    }

    /// Take the pending tiles out, for `take_refine_chunk` outside the lock
    /// (it encodes for some milliseconds). Give the rest back with
    /// [`Self::return_pending`].
    pub fn take_pending(&mut self) -> TileGrid {
        let (w, h) = self.pending.size();
        std::mem::replace(&mut self.pending, TileGrid::new(w, h))
    }

    /// The pending tiles that the last chunk did not take. When tiles remain,
    /// the next chunk is due at once.
    pub fn return_pending(&mut self, rest: &TileGrid, now: Instant) {
        if rest.size() == self.pending.size() {
            self.pending.union(rest);
        }
        self.due = (!self.pending.is_empty()).then_some(now);
    }

    /// The surface is about to be reset (live resize): a refinement frame
    /// taken before must not go out.
    pub fn drop_jobs(&mut self) {
        self.resets += 1;
    }

    /// Tag for a refinement frame taken now for surface `surface_id`.
    pub fn job_tag(&self, surface_id: u16) -> JobTag {
        JobTag {
            surface_id,
            resets: self.resets,
        }
    }

    /// Sequence number for the first bitmap of the next refinement frame.
    pub fn next_seq(&self) -> u8 {
        self.seq
    }

    /// A refinement frame `frame_id` with `count` bitmaps was queued.
    pub fn on_chunk_queued(&mut self, frame_id: u32, count: usize, now: Instant) {
        self.seq = self.seq.wrapping_add(count as u8);
        self.chunk_in_flight = Some((frame_id, now));
        self.clear_ids.push_back(frame_id);
        while self.clear_ids.len() > 16 {
            self.clear_ids.pop_front();
        }
    }

    /// The client acknowledged `frame_id` (acknowledgements come in order).
    pub fn on_ack(&mut self, frame_id: u32, suspended: bool) {
        if let Some((id, _)) = self.chunk_in_flight {
            if suspended || frame_id.wrapping_sub(id) < u32::MAX / 2 {
                self.chunk_in_flight = None;
            }
        }
    }

    /// Whether `frame_id` is a refinement frame.
    pub fn is_refine_frame(&self, frame_id: u32) -> bool {
        self.clear_ids.contains(&frame_id)
    }

    #[cfg(test)]
    fn pending(&self) -> &TileGrid {
        &self.pending
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Mark tiles in `g` from pixel rectangles, through `mark_changed` on two
    /// frames that differ in exactly those pixels.
    fn mark(g: &mut TileGrid, rects: &[(u32, u32, u32, u32)]) {
        let (w, h) = g.size();
        let stride = w as usize * 4;
        let prev = vec![0u8; stride * h as usize];
        let mut cur = prev.clone();
        for &(x, y, rw, rh) in rects {
            for yy in y..(y + rh).min(h) {
                for xx in x..(x + rw).min(w) {
                    cur[yy as usize * stride + xx as usize * 4] = 1;
                }
            }
        }
        g.mark_changed(&prev, &cur, stride);
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
    fn changed_pixels_mark_their_tiles() {
        let mut g = TileGrid::new(1280, 720);
        mark(&mut g, &[]);
        assert!(g.is_empty());
        mark(&mut g, &[(70, 10, 10, 10)]);
        assert_eq!(g.tiles().collect::<Vec<_>>(), [(2, 0)]);
        // Across a tile edge: two tiles.
        g.clear();
        mark(&mut g, &[(60, 0, 10, 1)]);
        assert_eq!(g.tiles().collect::<Vec<_>>(), [(1, 0), (2, 0)]);
        // A change that ends at a tile edge stays in its tile.
        g.clear();
        mark(&mut g, &[(64, 2, 32, 2)]);
        assert_eq!(g.tiles().collect::<Vec<_>>(), [(2, 0)]);
        // Last pixel of the screen.
        g.clear();
        mark(&mut g, &[(1279, 719, 1, 1)]);
        assert_eq!(g.tiles().collect::<Vec<_>>(), [(39, 21)]);
        // Tile rows of 32 and 33 pixels (720 / 22): row 21 starts at 687.
        g.clear();
        mark(&mut g, &[(0, 686, 1, 1)]);
        assert_eq!(g.tiles().collect::<Vec<_>>(), [(0, 20)]);
        // Marks add up.
        mark(&mut g, &[(0, 0, 1, 1)]);
        assert_eq!(g.tiles().count(), 2);
    }

    #[test]
    fn changed_pixels_with_row_padding_and_bad_input() {
        // Stride larger than the row: padding bytes do not count.
        let (w, h) = (64u32, 32u32);
        let stride = 64 * 4 + 32;
        let prev = vec![0u8; stride * h as usize];
        let mut cur = prev.clone();
        cur[5 * stride + 64 * 4 + 3] = 9; // padding
        let mut g = TileGrid::new(w, h);
        g.mark_changed(&prev, &cur, stride);
        assert!(g.is_empty());
        cur[5 * stride + 40 * 4] = 9; // pixel (40, 5)
        g.mark_changed(&prev, &cur, stride);
        assert_eq!(g.tiles().collect::<Vec<_>>(), [(1, 0)]);
        // A frame that is too short: everything.
        g.clear();
        g.mark_changed(&prev[..100], &cur, stride);
        assert!(g.is_full());
        // `diff_frames`: no usable previous frame, or another length: everything.
        assert!(diff_frames(&prev, &cur, stride, w, h, false).is_full());
        assert!(diff_frames(&prev[..100], &cur, stride, w, h, true).is_full());
        assert!(diff_frames(&cur, &cur, stride, w, h, true).is_empty());
    }

    #[test]
    fn rects_join_runs_in_a_row() {
        let mut g = TileGrid::new(1280, 720);
        mark(&mut g, &[(0, 0, 95, 1), (641, 700, 1, 1)]);
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

    #[test]
    fn scaled_pictures_up_or_down_use_the_scaled_share() {
        assert_eq!(color_share(1.0), COLOR_SHARE);
        assert_eq!(color_share(1.5), SCALED_COLOR_SHARE);
        assert_eq!(color_share(0.8), SCALED_COLOR_SHARE);
    }

    #[test]
    fn capture_scale_is_in_display_pixels() {
        // 1:1 capture of a 1x display.
        assert_eq!(
            color_share(capture_pixel_scale((1280, 720), (1280, 720), false)),
            COLOR_SHARE
        );
        // Retina at points: the capture scales 2880x1800 down to 1440x900.
        assert_eq!(
            color_share(capture_pixel_scale((1440, 900), (2880, 1800), false)),
            SCALED_COLOR_SHARE
        );
        // Client 1920x1080 on a 1280x720 display, letterboxed: 1.5x.
        assert_eq!(
            color_share(capture_pixel_scale((1920, 1080), (1280, 720), true)),
            SCALED_COLOR_SHARE
        );
        // Letterbox with bands on one axis only: the picture is 1:1.
        assert_eq!(
            color_share(capture_pixel_scale((1280, 960), (1280, 720), true)),
            COLOR_SHARE
        );
        // Stretched on one axis: scaled.
        assert_eq!(
            color_share(capture_pixel_scale((1280, 960), (1280, 720), false)),
            SCALED_COLOR_SHARE
        );
    }

    #[test]
    fn scaled_text_tile_with_more_colors_is_text() {
        // 32x32 tile with 300 colors: a picture at 1/4, text when scaled up (1/2).
        let data: Vec<u8> = (0..1024u32)
            .flat_map(|i| [(i % 300) as u8, (i % 300 / 256) as u8, 0, 255])
            .collect();
        let r = TileRect {
            x: 0,
            y: 0,
            w: 32,
            h: 32,
        };
        assert!(!is_text_like(&data, 128, r, COLOR_SHARE));
        assert!(is_text_like(&data, 128, r, SCALED_COLOR_SHARE));
    }

    /// A 128x64 BGRA frame (4x2 tiles of 32 px): `photo` tiles get a new color
    /// per pixel, the others two colors (text-like).
    fn refine_frame(photo: &[(u32, u32)]) -> Vec<u8> {
        let (w, h) = (128u32, 64u32);
        let mut data = vec![0u8; (w * h * 4) as usize];
        for y in 0..h {
            for x in 0..w {
                let i = ((y * w + x) * 4) as usize;
                let px = if photo.contains(&(x / 32, y / 32)) {
                    let v = y * w + x;
                    [v as u8, (v >> 8) as u8, (v * 7) as u8, 255]
                } else if (x + y) % 5 == 0 {
                    [0, 0, 0, 255]
                } else {
                    [255, 255, 255, 255]
                };
                data[i..i + 4].copy_from_slice(&px);
            }
        }
        data
    }

    fn tile(c: u16, r: u16, n: u16) -> TileRect {
        TileRect {
            x: c * 32,
            y: r * 32,
            w: n * 32,
            h: 32,
        }
    }

    #[test]
    fn refine_chunk_sends_text_and_skips_pictures() {
        let data = refine_frame(&[(1, 0)]);
        let mut pending = TileGrid::new(128, 64);
        pending.mark_all();
        let chunk = take_refine_chunk(&mut pending, &data, 128 * 4, COLOR_SHARE);
        assert!(pending.is_empty());
        assert_eq!((chunk.total, chunk.picked), (8, 7));
        let rects: Vec<TileRect> = chunk.tiles.iter().map(|(r, _)| *r).collect();
        // Row 0: tile 0 alone is a glyph-size run, so it takes the picture tile
        // next to it; tiles 2-3 are one run. Row 1: one run.
        assert_eq!(rects, vec![tile(0, 0, 2), tile(2, 0, 2), tile(0, 1, 4)]);
        let mut decoder = ironrdp_graphics::clearcodec::ClearCodecDecoder::new();
        for (r, bitmap) in &chunk.tiles {
            let px = decoder.decode(bitmap, r.w, r.h).unwrap();
            assert_eq!(px.len(), r.w as usize * r.h as usize * 4);
            // Lossless: the decoded colors are the source colors.
            let src = copy_rect(&data, 128 * 4, *r);
            let rgb = |b: &[u8]| -> Vec<[u8; 3]> {
                b.as_chunks::<4>()
                    .0
                    .iter()
                    .map(|p| [p[0], p[1], p[2]])
                    .collect()
            };
            assert_eq!(rgb(&px), rgb(&src));
        }
    }

    #[test]
    fn refine_chunk_stops_at_the_budget() {
        // A large text-like frame: several chunks, each of whole rows.
        let (w, h) = (1024u32, 1024u32);
        let data: Vec<u8> = (0..w * h)
            .flat_map(|i| {
                let v = if i.wrapping_mul(2_654_435_761) >> 30 == 0 {
                    0
                } else {
                    255
                };
                [v, v, v, 255]
            })
            .collect();
        let mut pending = TileGrid::new(w, h);
        pending.mark_all();
        let all = pending.tiles().count();
        let mut seen = 0;
        let mut chunks = 0;
        while !pending.is_empty() {
            let chunk = take_refine_chunk(&mut pending, &data, (w * 4) as usize, COLOR_SHARE);
            let bytes: usize = chunk.tiles.iter().map(|(_, d)| d.len()).sum();
            assert!(chunk.total.is_multiple_of(32), "whole rows");
            if !pending.is_empty() {
                assert!(bytes >= REFINE_CHUNK_BYTES);
            }
            seen += chunk.total;
            chunks += 1;
        }
        assert_eq!(seen, all);
        assert!(chunks > 1);
    }

    #[test]
    fn clear_seq_is_written_into_the_bitmaps() {
        let r = TileRect {
            x: 0,
            y: 0,
            w: 64,
            h: 32,
        };
        let mut tiles = vec![(r, vec![0x00, 0x00, 0xAA]), (r, vec![0x00, 0x00])];
        stamp_clear_seq(&mut tiles, 255);
        assert_eq!(tiles[0].1, vec![0x00, 255, 0xAA]);
        assert_eq!(tiles[1].1, vec![0x00, 0]);
    }

    #[test]
    fn compressed_pdu_decodes_after_any_history() {
        // One segment, exactly one full segment, and multipart messages.
        for len in [20_000usize, 65_535, 65_536, 150_000] {
            let pdu: Vec<u8> = (0..len).map(|i| ((i / 7) % 13) as u8).collect();
            let compressed = compress_wrapped(&zgfx::wrap_uncompressed(&pdu)).unwrap();
            assert!(
                compressed.len() < pdu.len() / 4,
                "{len}: {}",
                compressed.len()
            );
            // A client that already has other data in its history.
            let mut client = zgfx::Decompressor::new();
            let mut out = Vec::new();
            client
                .decompress(&zgfx::wrap_uncompressed(&[0x55; 3000]), &mut out)
                .unwrap();
            out.clear();
            client.decompress(&compressed, &mut out).unwrap();
            assert_eq!(out, pdu, "{len}");
        }
    }

    #[test]
    fn multipart_segments_hold_at_most_65535_bytes() {
        let pdu: Vec<u8> = (0..150_000usize).map(|i| (i % 251) as u8).collect();
        let c = compress_wrapped(&zgfx::wrap_uncompressed(&pdu)).unwrap();
        assert_eq!(c[0], 0xE1);
        let count = u16::from_le_bytes([c[1], c[2]]);
        assert_eq!(count, 3);
        assert_eq!(u32::from_le_bytes([c[3], c[4], c[5], c[6]]), 150_000);
        // Each segment alone decompresses to at most one segment of data.
        let mut pos = 7;
        let mut d = zgfx::Decompressor::new();
        let mut total = 0;
        for _ in 0..count {
            let size = u32::from_le_bytes(c[pos..pos + 4].try_into().unwrap()) as usize;
            let seg = &c[pos + 4..pos + 4 + size];
            let mut single = vec![0xE0];
            single.extend_from_slice(seg);
            let mut out = Vec::new();
            d.decompress(&single, &mut out).unwrap();
            assert!(out.len() <= 65_535);
            total += out.len();
            pos += 4 + size;
        }
        assert_eq!((pos, total), (c.len(), 150_000));
        // Data that does not compress stays raw.
        let mut x = 0x9E37_79B9_7F4A_7C15u64;
        let noise: Vec<u8> = (0..1000)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                (x >> 56) as u8
            })
            .collect();
        let c = compress_wrapped(&zgfx::wrap_uncompressed(&noise)).unwrap();
        assert_eq!(&c[..2], &[0xE0, 0x04]);
        assert_eq!(&c[2..], &noise[..]);
    }

    /// The capture side for a frame `cur` after `prev` (None: no previous frame).
    fn cap(r: &mut Refine, prev: Option<&[u8]>, cur: &[u8], w: u32, h: u32) {
        match prev {
            None => r.on_captured(None),
            Some(p) => {
                let g = diff_frames(p, cur, w as usize * 4, w, h, true);
                r.on_captured(Some(&g));
            }
        }
    }

    fn frame(w: u32, h: u32, v: u8) -> Vec<u8> {
        vec![v; (w * h * 4) as usize]
    }

    #[test]
    fn first_frame_covers_the_whole_surface() {
        let mut r = Refine::new(128, 64);
        let f = frame(128, 64, 0);
        cap(&mut r, None, &f, 128, 64);
        assert!(matches!(r.regions(false), Some(Regions::Whole)));
    }

    #[test]
    fn regions_are_changed_and_pending_tiles() {
        let mut r = Refine::new(128, 64);
        let a = frame(128, 64, 0);
        cap(&mut r, None, &a, 128, 64);
        let reg = r.regions(false).unwrap();
        r.on_encoded(0, reg);
        let reg = r.take_regions(0);
        r.on_shipped(&reg, Instant::now());
        // Everything is pending: a same frame repaints the pending tiles.
        assert!(matches!(r.regions(false), Some(Regions::Whole)));
        // Refine the whole screen.
        let now = Instant::now();
        assert_eq!(r.gate(true, true, 0, now), RefineGate::Go);
        let mut pending = r.take_pending();
        let chunk = take_refine_chunk(&mut pending, &a, 512, COLOR_SHARE);
        r.return_pending(&pending, now);
        assert_eq!(chunk.total, 8);
        assert!(r.pending().is_empty() && r.due().is_none());
        // Nothing changed: no frame.
        cap(&mut r, Some(&a), &a, 128, 64);
        assert!(r.regions(false).is_none());
        // One pixel in tile (1, 1).
        let mut b = a.clone();
        b[(40 * 128 + 40) * 4] = 7;
        cap(&mut r, Some(&a), &b, 128, 64);
        let Some(Regions::Tiles(g)) = r.regions(false) else {
            panic!("tiles")
        };
        assert_eq!(g.rects(), vec![tile(1, 1, 1)]);
        // A forced keyframe repaints all.
        assert!(matches!(r.regions(true), Some(Regions::Whole)));
    }

    #[test]
    fn no_compare_needed_while_every_tile_is_pending() {
        let mut r = Refine::new(128, 64);
        assert!(r.wants_changed_tiles());
        r.on_shipped(&Regions::Whole, Instant::now());
        assert!(!r.wants_changed_tiles());
        // A frame given without its tiles: whole, as it would be anyway.
        cap(&mut r, None, &frame(128, 64, 0), 128, 64);
        assert!(matches!(r.regions(false), Some(Regions::Whole)));
    }

    #[test]
    fn dropped_frames_keep_their_changes() {
        let mut r = Refine::new(128, 64);
        let a = frame(128, 64, 0);
        cap(&mut r, None, &a, 128, 64);
        r.on_encoded(0, Regions::Whole);
        r.take_regions(0);
        r.pending.clear();
        let mut b = a.clone();
        b[0] = 1; // tile (0, 0)
                  // Captured and dropped before the encoder: its changes stay dirty.
        cap(&mut r, Some(&a), &b, 128, 64);
        let mut c = b.clone();
        c[(40 * 128 + 100) * 4] = 1; // tile (3, 1)
        cap(&mut r, Some(&b), &c, 128, 64);
        let Some(Regions::Tiles(g)) = r.regions(false) else {
            panic!("tiles")
        };
        assert_eq!(g.tiles().collect::<Vec<_>>(), [(0, 0), (3, 1)]);
        // A dropped last frame stops the refinement until the next frame.
        assert_eq!(r.gate(true, true, 0, Instant::now()), RefineGate::Idle);
    }

    #[test]
    fn frames_the_encoder_skipped_add_their_regions() {
        let mut r = Refine::new(128, 64);
        let mut g1 = TileGrid::new(128, 64);
        let mut g2 = g1.clone();
        let one = |g: &mut TileGrid, x: u32| {
            let (w, h) = g.size();
            let prev = vec![0u8; (w * h * 4) as usize];
            let mut cur = prev.clone();
            cur[(x * 4) as usize] = 1;
            g.mark_changed(&prev, &cur, (w * 4) as usize);
        };
        one(&mut g1, 0);
        one(&mut g2, 100);
        r.on_encoded(5, Regions::Tiles(g1));
        r.on_encoded(6, Regions::Tiles(g2));
        // Frame 5 never came out of the encoder.
        let Regions::Tiles(g) = r.take_regions(6) else {
            panic!("tiles")
        };
        assert_eq!(g.tiles().collect::<Vec<_>>(), [(0, 0), (3, 0)]);
        // A frame with no entry: whole.
        assert!(matches!(r.take_regions(7), Regions::Whole));
    }

    #[test]
    fn a_refine_job_is_stale_after_any_surface_reset() {
        let mut r = Refine::new(128, 64);
        let tag = r.job_tag(3);
        assert_eq!(tag, r.job_tag(3));
        assert_ne!(tag, r.job_tag(4), "another surface");
        for reset in [
            Refine::drop_jobs as fn(&mut Refine),
            Refine::on_reset_graphics,
            Refine::on_new_encoder,
            |r: &mut Refine| r.on_new_surface(128, 64),
            |r: &mut Refine| r.on_new_surface(256, 64),
        ] {
            let before = r.job_tag(3);
            reset(&mut r);
            assert_ne!(before, r.job_tag(3));
        }
    }

    #[test]
    fn keyframes_repaint_the_whole_surface() {
        let g = TileGrid::new(128, 64);
        assert!(matches!(
            ship_regions(Regions::Tiles(g.clone()), false),
            Regions::Tiles(_)
        ));
        assert!(matches!(
            ship_regions(Regions::Tiles(g), true),
            Regions::Whole
        ));
    }

    #[test]
    fn new_surface_and_size_start_over() {
        let mut r = Refine::new(128, 64);
        r.on_encoded(0, Regions::Whole);
        r.on_chunk_queued(9, 3, Instant::now());
        r.on_new_surface(128, 64);
        assert!(matches!(r.regions(false), Some(Regions::Whole)));
        assert!(matches!(r.take_regions(0), Regions::Whole));
        // The ClearCodec sequence stays with the channel.
        assert_eq!(r.next_seq(), 3);
        r.on_new_surface(256, 64);
        assert_eq!(r.next_seq(), 3);
        // A frame of another size than the surface: all.
        let f = frame(64, 64, 0);
        cap(&mut r, Some(&f), &f, 64, 64);
        assert!(matches!(r.regions(false), Some(Regions::Whole)));
        r.on_reset_graphics();
        assert_eq!(r.next_seq(), 0);
    }

    #[test]
    fn sequence_numbers_follow_the_bitmaps_sent() {
        let mut r = Refine::new(128, 64);
        let now = Instant::now();
        r.on_chunk_queued(1, 255, now);
        r.on_chunk_queued(2, 2, now);
        assert_eq!(r.next_seq(), 1);
        assert!(r.is_refine_frame(1) && r.is_refine_frame(2) && !r.is_refine_frame(3));
    }

    #[test]
    fn refine_waits_for_frames_and_acks() {
        let mut r = Refine::new(128, 64);
        let t0 = Instant::now();
        r.on_unchanged();
        assert_eq!(r.gate(false, true, 0, t0), RefineGate::GiveUp);
        assert_eq!(r.gate(true, false, 0, t0), RefineGate::Idle);
        assert_eq!(r.gate(true, true, 1, t0), RefineGate::Wait);
        assert_eq!(r.gate(true, true, 0, t0), RefineGate::Go);
        // One refinement frame at a time.
        r.on_chunk_queued(10, 1, t0);
        assert_eq!(r.gate(true, true, 0, t0), RefineGate::Wait);
        // An ack of an older frame does not free it; of it or later does.
        r.on_ack(9, false);
        assert_eq!(r.gate(true, true, 0, t0), RefineGate::Wait);
        r.on_ack(11, false);
        assert_eq!(r.gate(true, true, 0, t0), RefineGate::Go);
        // Suspended acks, or no ack in time.
        r.on_chunk_queued(12, 1, t0);
        r.on_ack(0, true);
        assert_eq!(r.gate(true, true, 0, t0), RefineGate::Go);
        r.on_chunk_queued(13, 1, t0);
        assert_eq!(
            r.gate(true, true, 0, t0 + REFINE_ACK_TIMEOUT),
            RefineGate::Go
        );
    }

    #[test]
    fn refine_is_due_after_the_quiet_time() {
        let mut r = Refine::new(128, 64);
        assert!(r.due().is_none());
        let t0 = Instant::now();
        r.on_shipped(&Regions::Whole, t0);
        assert_eq!(r.due(), Some(t0 + REFINE_DELAY));
        r.poll_later(t0);
        assert_eq!(r.due(), Some(t0 + REFINE_POLL));
        r.give_up();
        assert!(r.due().is_none());
    }
}
