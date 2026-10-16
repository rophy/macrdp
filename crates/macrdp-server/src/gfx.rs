//! EGFX (MS-RDPEGFX) graphics pipeline for H.264 video.
//!
//! `MacGfxFactory` plugs ironrdp-egfx's `GraphicsPipelineServer` into
//! ironrdp-server. The server keeps one pipeline per connection and gives us a
//! shared handle to it; the display loop uses the free functions below to send
//! surfaces and H.264 frames through that handle.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};
use std::time::{Duration, Instant};

use ironrdp_dvc::{DvcEncode, DvcMessage};
use ironrdp_egfx::pdu::{
    Avc420Region, CapabilitiesAdvertisePdu, CapabilitiesV103Flags, CapabilitiesV104Flags,
    CapabilitiesV107Flags, CapabilitiesV10Flags, CapabilitySet,
};
use ironrdp_egfx::server::{
    FrameSubmissionError, GraphicsPipelineHandler, GraphicsPipelineServer, MixedTilePayload,
};
use ironrdp_graphics::zgfx::{self, CompressionMode};
use ironrdp_pdu::{Encode, EncodeResult, WriteCursor};
use ironrdp_server::{
    EgfxServerMessage, GfxDvcBridge, GfxServerFactory, GfxServerHandle, ServerEvent,
    ServerEventSender,
};
use ironrdp_svc::ChannelFlags;
use tokio::sync::mpsc;

use crate::frame_window::{FrameKind, FrameWindow};
use crate::refine::TileRect;
use ironrdp_pdu::geometry::ExclusiveRectangle;

/// H.264 quantization parameter and quality written into each AVC420 region.
const REGION_QP: u8 = 22;
const REGION_QUALITY: u8 = 100;

/// State shared between the EGFX factory/handler and the display loop.
#[derive(Default)]
pub struct GfxShared {
    /// Pipeline of the current connection.
    pub handle: Mutex<Option<GfxServerHandle>>,
    /// Server event sender, used to put EGFX PDUs on the wire.
    pub sender: Mutex<Option<mpsc::UnboundedSender<ServerEvent>>>,
    /// Capabilities are negotiated and the channel is open.
    pub ready: AtomicBool,
    /// The negotiated capabilities allow AVC444.
    pub avc444: AtomicBool,
    /// Incremented on every `on_ready` (new or re-negotiated channel).
    pub generation: AtomicU64,
    /// Surface the display loop draws on. Kept here and not in the display
    /// loop: a resize ends one display loop and starts the next one on the
    /// same EGFX channel, which still has this surface.
    pub surface: Mutex<Option<SurfaceInfo>>,
    /// EGFX frame window of the current channel. Lock order: pipeline, then this.
    pub window: Mutex<FrameWindow>,
    /// ClearCodec sequence number the client expects next. Lock order: pipeline, then this.
    clear_seq: Mutex<ClearSeq>,
}

/// ClearCodec sequence numbers (MS-RDPEGFX 2.2.4.1 `seqNumber`).
///
/// The client keeps one ClearCodec context for the connection, not one per
/// surface or per display loop, and sets its counter back to 0 only on
/// ResetGraphics (FreeRDP: `clear_context_reset` from `gdi_ResetGraphics`).
/// It refuses a bitmap with another number. So the number belongs to the EGFX
/// channel; it starts at 0 on each generation (the pipeline sends
/// ResetGraphics before the first surface of a generation) and after each
/// ResetGraphics, and it moves on only for bitmaps that went out.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
pub struct ClearSeq {
    /// (generation, next number)
    next: Option<(u64, u8)>,
}

impl ClearSeq {
    /// The number for the next bitmap on channel `generation`.
    pub fn next(&self, generation: u64) -> u8 {
        match self.next {
            Some((g, n)) if g == generation => n,
            _ => 0,
        }
    }

    /// `count` bitmaps went out on channel `generation`.
    pub fn sent(&mut self, generation: u64, count: usize) {
        let n = self.next(generation).wrapping_add(count as u8);
        self.next = Some((generation, n));
    }

    /// ResetGraphics went out on channel `generation`.
    pub fn reset(&mut self, generation: u64) {
        self.next = Some((generation, 0));
    }
}

/// Write sequence numbers from `first` into ClearCodec bitmap streams
/// (byte 0 is the flags, byte 1 the sequence number).
fn stamp_clear_seq(tiles: &mut [(TileRect, Vec<u8>)], first: u8) {
    for (i, (_, data)) in tiles.iter_mut().enumerate() {
        if let Some(seq) = data.get_mut(1) {
            *seq = first.wrapping_add(i as u8);
        }
    }
}

/// An EGFX surface that was created and mapped to the output.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SurfaceInfo {
    pub id: u16,
    /// EGFX generation (channel) the surface belongs to
    pub generation: u64,
    pub width: u16,
    pub height: u16,
}

/// What to do before a frame of a given size can be sent.
#[derive(Debug, PartialEq, Eq)]
pub enum SurfacePlan {
    /// The surface fits: send on it.
    Reuse(u16),
    /// Create a surface. If ResetGraphics was not sent on this channel, the
    /// pipeline sends it first, at the surface size.
    Create,
    /// Send ResetGraphics for the new size (this deletes the old surface),
    /// then create a surface.
    ResizeThenCreate,
}

/// Decide how to get a surface for a `size` frame on channel `generation`.
///
/// `output` is the pipeline's graphics output size ((0, 0) before the first
/// ResetGraphics on the channel). A channel that re-negotiated its
/// capabilities drops its surfaces but keeps the output size; a new surface
/// of another size then needs a ResetGraphics too.
pub fn plan_surface(
    prev: Option<SurfaceInfo>,
    generation: u64,
    size: (u16, u16),
    output: (u16, u16),
) -> SurfacePlan {
    match prev {
        Some(s) if s.generation == generation => {
            if (s.width, s.height) == size {
                SurfacePlan::Reuse(s.id)
            } else {
                SurfacePlan::ResizeThenCreate
            }
        }
        _ if output != (0, 0) && output != size => SurfacePlan::ResizeThenCreate,
        _ => SurfacePlan::Create,
    }
}

pub type SharedGfx = Arc<GfxShared>;

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(PoisonError::into_inner)
}

/// Whether a negotiated capability set allows AVC444.
///
/// Same rule as ironrdp-egfx's codec capabilities: V10 and later allow AVC420
/// and AVC444 unless `AVC_DISABLED` is set (V10.1 has no flags). V8.1 allows
/// AVC420 only and V8 allows no AVC.
fn caps_allow_avc444(caps: &CapabilitySet) -> bool {
    match caps {
        CapabilitySet::V8 { .. } | CapabilitySet::V8_1 { .. } => false,
        CapabilitySet::V10 { flags } | CapabilitySet::V10_2 { flags } => {
            !flags.contains(CapabilitiesV10Flags::AVC_DISABLED)
        }
        CapabilitySet::V10_1 => true,
        CapabilitySet::V10_3 { flags } => !flags.contains(CapabilitiesV103Flags::AVC_DISABLED),
        CapabilitySet::V10_4 { flags }
        | CapabilitySet::V10_5 { flags }
        | CapabilitySet::V10_6 { flags }
        | CapabilitySet::V10_6Err { flags } => !flags.contains(CapabilitiesV104Flags::AVC_DISABLED),
        CapabilitySet::V10_7 { flags } => !flags.contains(CapabilitiesV107Flags::AVC_DISABLED),
    }
}

struct MacGfxHandler {
    shared: SharedGfx,
}

impl GraphicsPipelineHandler for MacGfxHandler {
    fn capabilities_advertise(&mut self, _pdu: &CapabilitiesAdvertisePdu) {}

    fn on_ready(&mut self, negotiated: &CapabilitySet) {
        // Called while the DVC bridge holds the pipeline mutex: do not lock
        // `shared.handle`'s server here.
        let avc444 = caps_allow_avc444(negotiated);
        tracing::info!(?negotiated, avc444, "EGFX channel ready");
        self.shared.avc444.store(avc444, Ordering::SeqCst);
        self.shared.generation.fetch_add(1, Ordering::SeqCst);
        lock(&self.shared.window).reset();
        self.shared.ready.store(true, Ordering::SeqCst);
    }

    fn on_frame_ack(&mut self, frame_id: u32, queue_depth: u32, _total_frames_decoded: u32) {
        // Called with the pipeline mutex held (lock order: pipeline, window).
        let rtt = lock(&self.shared.window).on_ack(frame_id, queue_depth, Instant::now());
        tracing::trace!(frame_id, queue_depth, ?rtt, "EGFX frame acknowledged");
    }

    fn on_close(&mut self) {
        tracing::info!("EGFX channel closed");
        self.shared.ready.store(false, Ordering::SeqCst);
    }
}

pub struct MacGfxFactory {
    shared: SharedGfx,
}

impl MacGfxFactory {
    pub fn new(shared: SharedGfx) -> Self {
        Self { shared }
    }
}

impl ServerEventSender for MacGfxFactory {
    fn set_sender(&mut self, sender: mpsc::UnboundedSender<ServerEvent>) {
        *lock(&self.shared.sender) = Some(sender);
    }
}

impl GfxServerFactory for MacGfxFactory {
    fn build_gfx_handler(&self) -> Box<dyn GraphicsPipelineHandler> {
        Box::new(MacGfxHandler {
            shared: Arc::clone(&self.shared),
        })
    }

    fn build_server_with_handle(&self) -> Option<(GfxDvcBridge, GfxServerHandle)> {
        // No ZGFX compression. Upstream `compress_and_wrap_egfx` checks only the
        // compressed size against the 65535-byte single-segment limit, not the
        // input size: a 64-70 KiB PDU that compresses a little goes out as one
        // oversized segment and FreeRDP fails with `zgfx_decompress failure`.
        // H.264 payloads do not compress anyway, and with `Never` the server
        // never back-references the ZGFX history, so ordering between inline
        // replies and queued frames cannot desync the client. Lossless
        // refinement frames are compressed on their own (`flush_compressed`).
        let server = GraphicsPipelineServer::with_compression(
            self.build_gfx_handler(),
            CompressionMode::Never,
        );
        let handle: GfxServerHandle = Arc::new(Mutex::new(server));
        // A new connection: the previous channel (if any) is gone.
        self.shared.ready.store(false, Ordering::SeqCst);
        self.shared.avc444.store(false, Ordering::SeqCst);
        *lock(&self.shared.surface) = None;
        lock(&self.shared.window).reset();
        *lock(&self.shared.clear_seq) = ClearSeq::default();
        *lock(&self.shared.handle) = Some(Arc::clone(&handle));
        Some((GfxDvcBridge::new(Arc::clone(&handle)), handle))
    }
}

/// Drain the queued EGFX PDUs and hand them to the server event loop.
fn flush(
    shared: &GfxShared,
    server: &mut GraphicsPipelineServer,
) -> Result<(), FrameSubmissionError> {
    let msgs = server.drain_output();
    send_messages(shared, server, msgs)
}

/// Like `flush`, but each PDU is ZGFX-compressed on its own.
///
/// The pipeline sends PDUs uncompressed (see `build_server_with_handle`). A
/// PDU compressed with a new compressor refers only to its own bytes, so the
/// client decodes it correctly whatever else it put into its ZGFX history
/// before, and in whatever order the other messages reach it.
/// Returns the size of the compressed messages.
fn flush_compressed(
    shared: &GfxShared,
    server: &mut GraphicsPipelineServer,
) -> Result<usize, FrameSubmissionError> {
    let msgs = server
        .drain_output()
        .into_iter()
        .map(compress_message)
        .collect::<Result<Vec<_>, _>>()?;
    let size = msgs.iter().map(|m| m.size()).sum();
    send_messages(shared, server, msgs)?;
    Ok(size)
}

/// Compress one ZGFX-wrapped (uncompressed) EGFX message on its own.
fn compress_message(msg: DvcMessage) -> Result<DvcMessage, FrameSubmissionError> {
    let fail = |what: &str, e: &dyn std::fmt::Display| {
        tracing::warn!(error = %e, "EGFX {what} failed");
        FrameSubmissionError::InvalidFrame
    };
    let mut wrapped = vec![0u8; msg.size()];
    msg.encode(&mut WriteCursor::new(&mut wrapped))
        .map_err(|e| fail("PDU encode", &e))?;
    // The wrapped PDU has raw segments only: a new decompressor unwraps it.
    let mut pdu = Vec::with_capacity(wrapped.len());
    zgfx::Decompressor::new()
        .decompress(&wrapped, &mut pdu)
        .map_err(|e| fail("ZGFX unwrap", &e))?;
    let bytes =
        zgfx::compress_and_wrap_egfx(&pdu, &mut zgfx::Compressor::new(), CompressionMode::Auto)
            .map_err(|e| fail("ZGFX compress", &e))?;
    Ok(Box::new(ZgfxMessage {
        bytes,
        name: msg.name(),
    }))
}

/// A ZGFX-wrapped EGFX PDU.
struct ZgfxMessage {
    bytes: Vec<u8>,
    name: &'static str,
}

impl Encode for ZgfxMessage {
    fn encode(&self, dst: &mut WriteCursor<'_>) -> EncodeResult<()> {
        dst.write_slice(&self.bytes);
        Ok(())
    }

    fn name(&self) -> &'static str {
        self.name
    }

    fn size(&self) -> usize {
        self.bytes.len()
    }
}

impl DvcEncode for ZgfxMessage {}

/// Hand drained EGFX messages to the server event loop.
fn send_messages(
    shared: &GfxShared,
    server: &GraphicsPipelineServer,
    msgs: Vec<DvcMessage>,
) -> Result<(), FrameSubmissionError> {
    if msgs.is_empty() {
        return Ok(());
    }
    let channel_id = server.channel_id().ok_or(FrameSubmissionError::NotReady)?;
    let messages = ironrdp_dvc::encode_dvc_messages(channel_id, msgs, ChannelFlags::empty())
        .map_err(|e| {
            tracing::warn!(error = %e, "EGFX DVC encode failed");
            FrameSubmissionError::InvalidFrame
        })?;
    let sender = lock(&shared.sender);
    let sender = sender.as_ref().ok_or(FrameSubmissionError::NotReady)?;
    sender
        .send(ServerEvent::Egfx(EgfxServerMessage::SendMessages {
            messages,
        }))
        .map_err(|_| FrameSubmissionError::NotReady)
}

fn current_handle(shared: &GfxShared) -> Option<GfxServerHandle> {
    lock(&shared.handle).clone()
}

fn full_region(w: u16, h: u16) -> [Avc420Region; 1] {
    // Right and bottom are exclusive.
    [Avc420Region::new(0, 0, w, h, REGION_QP, REGION_QUALITY)]
}

/// Queue one AVC420 frame and send it to the wire. Returns Err if the frame was not accepted.
/// `regions` are the parts of the decoded frame that the client copies to the
/// surface (MS-RDPEGFX 2.2.4.4.1); empty means the whole frame.
pub fn submit_avc420(
    shared: &GfxShared,
    surface: u16,
    data: &[u8],
    regions: &[TileRect],
    w: u16,
    h: u16,
    ts_ms: u32,
) -> Result<(), FrameSubmissionError> {
    let handle = current_handle(shared).ok_or(FrameSubmissionError::NotReady)?;
    let mut server = lock(&handle);
    let regions: Vec<Avc420Region> = if regions.is_empty() {
        full_region(w, h).to_vec()
    } else {
        regions
            .iter()
            .map(|r| Avc420Region::new(r.x, r.y, r.x + r.w, r.y + r.h, REGION_QP, REGION_QUALITY))
            .collect()
    };
    let frame_id = server.try_send_avc420_frame(surface, data, &regions, ts_ms)?;
    flush(shared, &mut server)?;
    lock(&shared.window).on_sent(frame_id, FrameKind::Video, Instant::now());
    Ok(())
}

/// Queue one frame of ClearCodec (lossless) tiles and send it to the wire.
/// The tiles get their sequence numbers here. Returns the number of bytes queued.
pub fn submit_clearcodec(
    shared: &GfxShared,
    surface: SurfaceInfo,
    mut tiles: Vec<(TileRect, Vec<u8>)>,
    ts_ms: u32,
) -> Result<usize, FrameSubmissionError> {
    let handle = current_handle(shared).ok_or(FrameSubmissionError::NotReady)?;
    let mut server = lock(&handle);
    let count = tiles.len();
    stamp_clear_seq(&mut tiles, lock(&shared.clear_seq).next(surface.generation));
    let bytes = tiles.iter().map(|(_, d)| d.len()).sum();
    let tiles = tiles
        .into_iter()
        .map(|(r, bitmap_data)| MixedTilePayload::ClearCodec {
            destination: ExclusiveRectangle {
                left: r.x,
                top: r.y,
                right: r.x + r.w,
                bottom: r.y + r.h,
            },
            bitmap_data,
        })
        .collect();
    let frame_id = server
        .send_mixed_frame(surface.id, tiles, ts_ms)
        .ok_or(FrameSubmissionError::InvalidFrame)?;
    // ClearCodec residual data compresses well; H.264 does not.
    let t0 = Instant::now();
    let wire = flush_compressed(shared, &mut server)?;
    lock(&shared.clear_seq).sent(surface.generation, count);
    lock(&shared.window).on_sent(frame_id, FrameKind::Other, Instant::now());
    tracing::debug!(
        frame_id,
        bytes,
        wire,
        zgfx_ms = format!("{:.1}", t0.elapsed().as_secs_f64() * 1000.0),
        "EGFX ClearCodec frame sent"
    );
    Ok(bytes)
}

/// Queue one AVC444 frame (luma + chroma streams) and send it to the wire.
/// An empty `chroma` sends a luma-only frame.
pub fn submit_avc444(
    shared: &GfxShared,
    surface: u16,
    luma: &[u8],
    chroma: &[u8],
    w: u16,
    h: u16,
    ts_ms: u32,
) -> Result<(), FrameSubmissionError> {
    let handle = current_handle(shared).ok_or(FrameSubmissionError::NotReady)?;
    let mut server = lock(&handle);
    let regions = full_region(w, h);
    let (chroma_data, chroma_regions) = if chroma.is_empty() {
        (None, None)
    } else {
        (Some(chroma), Some(&regions[..]))
    };
    let frame_id = server.try_send_avc444_frame(
        surface,
        luma,
        &regions,
        chroma_data,
        chroma_regions,
        ts_ms,
    )?;
    flush(shared, &mut server)?;
    lock(&shared.window).on_sent(frame_id, FrameKind::Video, Instant::now());
    Ok(())
}

/// Ok(()) if a frame could be accepted now (call before encoding).
/// `frame_interval` is the capture interval; it sets the frame window.
pub fn can_submit(
    shared: &GfxShared,
    surface: u16,
    avc444: bool,
    frame_interval: Duration,
) -> Result<(), FrameSubmissionError> {
    let handle = current_handle(shared).ok_or(FrameSubmissionError::NotReady)?;
    let mut server = lock(&handle);
    apply_window(shared, &mut server, frame_interval);
    if avc444 {
        server.avc444_submission_state(surface)
    } else {
        server.avc420_submission_state(surface)
    }
}

/// Set the pipeline's frame window from `FrameWindow`.
///
/// ironrdp-server sets the client's core Frame Acknowledge value (mstsc: 1) as
/// a ceiling on the EGFX window after each capability exchange. That value is
/// for surface-command frames (MS-RDPRFX 2.2.1.3), not EGFX, so it is removed here.
fn apply_window(shared: &GfxShared, server: &mut GraphicsPipelineServer, frame_interval: Duration) {
    let want = lock(&shared.window).target(frame_interval);
    let had = server.max_frames_in_flight();
    if had != want {
        server.set_client_frame_ack_limit(None);
        server.set_max_frames_in_flight(want);
        tracing::debug!(from = had, to = want, "EGFX frame window");
    }
}

/// Current frame window, base RTT (ms) and last client queue depth, for logs.
pub fn window_stats(shared: &GfxShared) -> (u32, Option<f32>, u32) {
    let window = current_handle(shared)
        .map(|h| lock(&h).max_frames_in_flight())
        .unwrap_or(0);
    let w = lock(&shared.window);
    let base = w.base_rtt().map(|d| d.as_secs_f32() * 1000.0);
    (window, base, w.last_queue_depth)
}

/// Whether the current connection's EGFX channel allows AVC420 (H.264).
/// None while that channel is not open. A client can negotiate EGFX with
/// `AVC_DISABLED`; then video must use bitmap updates.
pub fn avc420_allowed(shared: &GfxShared) -> Option<bool> {
    let handle = current_handle(shared)?;
    let server = lock(&handle);
    server.is_ready().then(|| server.supports_avc420())
}

/// Get the surface for a w×h frame on channel `generation`: the current one
/// if it fits, else a new one (after ResetGraphics when the size changed).
/// Returns (surface id, true if the surface is new); a new surface needs a keyframe.
pub fn ensure_surface(shared: &GfxShared, generation: u64, w: u16, h: u16) -> Option<(u16, bool)> {
    let handle = current_handle(shared)?;
    let mut server = lock(&handle);
    let mut slot = lock(&shared.surface);
    match plan_surface(*slot, generation, (w, h), server.output_dimensions()) {
        SurfacePlan::Reuse(id) => return Some((id, false)),
        SurfacePlan::ResizeThenCreate => {
            tracing::info!(
                old = ?server.output_dimensions(),
                width = w,
                height = h,
                "EGFX ResetGraphics for new desktop size"
            );
            server.resize(w, h);
            lock(&shared.clear_seq).reset(generation);
        }
        SurfacePlan::Create => {}
    }
    *slot = None;
    let id = server.create_surface(w, h)?;
    server.map_surface_to_output(id, 0, 0);
    if let Err(e) = flush(shared, &mut server) {
        tracing::warn!(error = %e, "EGFX surface PDUs not sent");
        return None;
    }
    *slot = Some(SurfaceInfo {
        id,
        generation,
        width: w,
        height: h,
    });
    Some((id, true))
}

/// EGFX frames sent and not acknowledged yet.
pub fn frames_in_flight(shared: &GfxShared) -> u32 {
    current_handle(shared)
        .map(|h| lock(&h).frames_in_flight())
        .unwrap_or(0)
}

/// The surface that frames are drawn on now, if any.
pub fn current_surface(shared: &GfxShared) -> Option<SurfaceInfo> {
    *lock(&shared.surface)
}

/// The pipeline no longer has our surface: create a new one with the next frame.
pub fn forget_surface(shared: &GfxShared) {
    *lock(&shared.surface) = None;
}

/// Median queueing delay of the stream frames acknowledged in the last `span`
/// and the number of samples (see `FrameWindow::queue_delay`).
pub fn queue_delay(shared: &GfxShared, span: Duration) -> Option<(Duration, usize)> {
    lock(&shared.window).queue_delay(Instant::now(), span)
}

/// Average frame round-trip time from frame acknowledgements, if measured.
pub fn rtt_ms(shared: &GfxShared) -> Option<f32> {
    let handle = current_handle(shared)?;
    let server = lock(&handle);
    server
        .qoe_snapshot()
        .filter(|s| s.total_rtt_samples > 0)
        .map(|s| s.avg_rtt_ms)
}

#[cfg(test)]
mod tests {
    use super::*;
    use ironrdp_egfx::pdu::{CapabilitiesV81Flags, CapabilitiesV8Flags};

    fn wire_bytes(msg: &DvcMessage) -> Vec<u8> {
        let mut out = vec![0u8; msg.size()];
        msg.encode(&mut WriteCursor::new(&mut out)).unwrap();
        out
    }

    #[test]
    fn compressed_message_decodes_after_any_history() {
        // One segment and a multipart message (more than 65535 bytes).
        for len in [20_000usize, 150_000] {
            let pdu: Vec<u8> = (0..len).map(|i| ((i / 7) % 13) as u8).collect();
            let msg: DvcMessage = Box::new(ZgfxMessage {
                bytes: zgfx::wrap_uncompressed(&pdu),
                name: "test",
            });
            let compressed = wire_bytes(&compress_message(msg).unwrap());
            assert!(compressed.len() < pdu.len() / 4, "{}", compressed.len());
            // A client that already has other data in its history.
            let mut client = zgfx::Decompressor::new();
            let mut out = Vec::new();
            let other = zgfx::wrap_uncompressed(&[0x55; 3000]);
            client.decompress(&other, &mut out).unwrap();
            out.clear();
            client.decompress(&compressed, &mut out).unwrap();
            assert_eq!(out, pdu);
        }
    }

    #[test]
    fn clear_seq_belongs_to_the_channel() {
        let mut seq = ClearSeq::default();
        assert_eq!(seq.next(1), 0);
        seq.sent(1, 3);
        // A new display loop on the same channel (resize to the same size, or
        // a new surface without ResetGraphics) goes on from the client's number.
        assert_eq!(seq.next(1), 3);
        // ResetGraphics sets the client back to 0.
        seq.reset(1);
        assert_eq!(seq.next(1), 0);
        seq.sent(1, 255);
        seq.sent(1, 2);
        assert_eq!(seq.next(1), 1);
        // A new channel starts with ResetGraphics.
        assert_eq!(seq.next(2), 0);
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
    fn failed_clearcodec_submit_keeps_the_sequence() {
        // The pipeline refuses the frame (not ready): no number is used up.
        let shared = Arc::new(GfxShared::default());
        let factory = MacGfxFactory::new(Arc::clone(&shared));
        let (_bridge, _handle) = factory.build_server_with_handle().unwrap();
        let tile = (
            TileRect {
                x: 0,
                y: 0,
                w: 64,
                h: 32,
            },
            vec![0, 0],
        );
        let surface = SurfaceInfo { generation: 1, ..S };
        assert!(submit_clearcodec(&shared, surface, vec![tile], 0).is_err());
        assert_eq!(shared.clear_seq.lock().unwrap().next(1), 0);
    }

    #[test]
    fn cannot_submit_before_ready() {
        let shared = Arc::new(GfxShared::default());
        let factory = MacGfxFactory::new(Arc::clone(&shared));
        let (_bridge, _handle) = factory.build_server_with_handle().unwrap();
        assert!(matches!(
            can_submit(&shared, 0, false, Duration::from_millis(16)),
            Err(FrameSubmissionError::NotReady)
        ));
    }

    #[test]
    fn cannot_submit_without_connection() {
        let shared = GfxShared::default();
        assert!(matches!(
            can_submit(&shared, 0, false, Duration::from_millis(16)),
            Err(FrameSubmissionError::NotReady)
        ));
    }

    #[test]
    fn factory_publishes_handle_per_connection() {
        let shared = Arc::new(GfxShared::default());
        let factory = MacGfxFactory::new(Arc::clone(&shared));
        let (_b1, h1) = factory.build_server_with_handle().unwrap();
        let (_b2, h2) = factory.build_server_with_handle().unwrap();
        let current = shared.handle.lock().unwrap().clone().unwrap();
        assert!(Arc::ptr_eq(&current, &h2) && !Arc::ptr_eq(&h1, &h2));
    }

    #[test]
    fn new_connection_clears_ready() {
        let shared = Arc::new(GfxShared::default());
        shared.ready.store(true, Ordering::SeqCst);
        let factory = MacGfxFactory::new(Arc::clone(&shared));
        let _ = factory.build_server_with_handle().unwrap();
        assert!(!shared.ready.load(Ordering::SeqCst));
    }

    const S: SurfaceInfo = SurfaceInfo {
        id: 3,
        generation: 7,
        width: 1024,
        height: 720,
    };

    #[test]
    fn surface_first_on_new_channel() {
        assert_eq!(
            plan_surface(None, 7, (1024, 720), (0, 0)),
            SurfacePlan::Create
        );
        // A surface of an older channel is gone with that channel.
        assert_eq!(
            plan_surface(Some(S), 8, (1024, 720), (0, 0)),
            SurfacePlan::Create
        );
        assert_eq!(
            plan_surface(Some(S), 8, (800, 600), (0, 0)),
            SurfacePlan::Create
        );
    }

    #[test]
    fn surface_reused_at_same_size() {
        // Also after a deactivation-reactivation that keeps the channel.
        assert_eq!(
            plan_surface(Some(S), 7, (1024, 720), (1024, 720)),
            SurfacePlan::Reuse(3)
        );
    }

    #[test]
    fn surface_resize_on_same_channel() {
        assert_eq!(
            plan_surface(Some(S), 7, (1281, 801), (1024, 720)),
            SurfacePlan::ResizeThenCreate
        );
        assert_eq!(
            plan_surface(Some(S), 7, (1024, 721), (1024, 720)),
            SurfacePlan::ResizeThenCreate
        );
    }

    #[test]
    fn surface_after_renegotiation_keeps_output_size() {
        // Re-negotiated channel: no surface, but the output has the old size.
        assert_eq!(
            plan_surface(None, 8, (1024, 720), (1024, 720)),
            SurfacePlan::Create
        );
        assert_eq!(
            plan_surface(Some(S), 8, (800, 600), (1024, 720)),
            SurfacePlan::ResizeThenCreate
        );
        assert_eq!(
            plan_surface(None, 8, (800, 600), (1024, 720)),
            SurfacePlan::ResizeThenCreate
        );
    }

    #[test]
    fn new_connection_clears_surface() {
        let shared = Arc::new(GfxShared::default());
        *shared.surface.lock().unwrap() = Some(S);
        let factory = MacGfxFactory::new(Arc::clone(&shared));
        let _ = factory.build_server_with_handle().unwrap();
        assert!(shared.surface.lock().unwrap().is_none());
    }

    #[test]
    fn avc444_v8_no() {
        assert!(!caps_allow_avc444(&CapabilitySet::V8 {
            flags: CapabilitiesV8Flags::empty()
        }));
    }

    #[test]
    fn avc444_v8_1_avc420_only() {
        assert!(!caps_allow_avc444(&CapabilitySet::V8_1 {
            flags: CapabilitiesV81Flags::AVC420_ENABLED
        }));
    }

    #[test]
    fn avc444_v10_and_v10_2() {
        for set in [
            |flags| CapabilitySet::V10 { flags },
            |flags| CapabilitySet::V10_2 { flags },
        ] {
            assert!(caps_allow_avc444(&set(CapabilitiesV10Flags::empty())));
            assert!(!caps_allow_avc444(&set(CapabilitiesV10Flags::AVC_DISABLED)));
        }
    }

    #[test]
    fn avc444_v10_1() {
        assert!(caps_allow_avc444(&CapabilitySet::V10_1));
    }

    #[test]
    fn avc444_v10_3() {
        assert!(caps_allow_avc444(&CapabilitySet::V10_3 {
            flags: CapabilitiesV103Flags::empty()
        }));
        assert!(!caps_allow_avc444(&CapabilitySet::V10_3 {
            flags: CapabilitiesV103Flags::AVC_DISABLED
        }));
    }

    #[test]
    fn avc444_v10_4_to_v10_6err() {
        for set in [
            |flags| CapabilitySet::V10_4 { flags },
            |flags| CapabilitySet::V10_5 { flags },
            |flags| CapabilitySet::V10_6 { flags },
            |flags| CapabilitySet::V10_6Err { flags },
        ] {
            assert!(caps_allow_avc444(&set(CapabilitiesV104Flags::empty())));
            assert!(!caps_allow_avc444(&set(
                CapabilitiesV104Flags::AVC_DISABLED
            )));
        }
    }

    #[test]
    fn avc444_v10_7() {
        assert!(caps_allow_avc444(&CapabilitySet::V10_7 {
            flags: CapabilitiesV107Flags::empty()
        }));
        assert!(!caps_allow_avc444(&CapabilitySet::V10_7 {
            flags: CapabilitiesV107Flags::AVC_DISABLED
        }));
    }
}
