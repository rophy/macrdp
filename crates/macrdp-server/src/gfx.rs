//! EGFX (MS-RDPEGFX) graphics pipeline for H.264 video.
//!
//! `MacGfxFactory` plugs ironrdp-egfx's `GraphicsPipelineServer` into
//! ironrdp-server. The server keeps one pipeline per connection and gives us a
//! shared handle to it; the display loop uses the free functions below to send
//! surfaces and H.264 frames through that handle.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use ironrdp_egfx::pdu::{
    Avc420Region, CapabilitiesAdvertisePdu, CapabilitiesV103Flags, CapabilitiesV104Flags,
    CapabilitiesV107Flags, CapabilitiesV10Flags, CapabilitySet,
};
use ironrdp_egfx::server::{FrameSubmissionError, GraphicsPipelineHandler, GraphicsPipelineServer};
use ironrdp_graphics::zgfx::CompressionMode;
use ironrdp_server::{
    EgfxServerMessage, GfxDvcBridge, GfxServerFactory, GfxServerHandle, ServerEvent,
    ServerEventSender,
};
use ironrdp_svc::ChannelFlags;
use tokio::sync::mpsc;

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
        self.shared.ready.store(true, Ordering::SeqCst);
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
        let server = GraphicsPipelineServer::with_compression(
            self.build_gfx_handler(),
            CompressionMode::Auto,
        );
        let handle: GfxServerHandle = Arc::new(Mutex::new(server));
        // A new connection: the previous channel (if any) is gone.
        self.shared.ready.store(false, Ordering::SeqCst);
        self.shared.avc444.store(false, Ordering::SeqCst);
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
pub fn submit_avc420(
    shared: &GfxShared,
    surface: u16,
    data: &[u8],
    w: u16,
    h: u16,
    ts_ms: u32,
) -> Result<(), FrameSubmissionError> {
    let handle = current_handle(shared).ok_or(FrameSubmissionError::NotReady)?;
    let mut server = lock(&handle);
    server.try_send_avc420_frame(surface, data, &full_region(w, h), ts_ms)?;
    flush(shared, &mut server)
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
    server.try_send_avc444_frame(surface, luma, &regions, chroma_data, chroma_regions, ts_ms)?;
    flush(shared, &mut server)
}

/// Ok(()) if a frame could be accepted now (call before encoding).
pub fn can_submit(
    shared: &GfxShared,
    surface: u16,
    avc444: bool,
) -> Result<(), FrameSubmissionError> {
    let handle = current_handle(shared).ok_or(FrameSubmissionError::NotReady)?;
    let server = lock(&handle);
    if avc444 {
        server.avc444_submission_state(surface)
    } else {
        server.avc420_submission_state(surface)
    }
}

/// Whether the current connection's EGFX channel allows AVC420 (H.264).
/// None while that channel is not open. A client can negotiate EGFX with
/// `AVC_DISABLED`; then video must use bitmap updates.
pub fn avc420_allowed(shared: &GfxShared) -> Option<bool> {
    let handle = current_handle(shared)?;
    let server = lock(&handle);
    server.is_ready().then(|| server.supports_avc420())
}

/// Create a w×h surface and map it to the output at (0, 0); returns the surface id.
pub fn create_surface(shared: &GfxShared, w: u16, h: u16) -> Option<u16> {
    let handle = current_handle(shared)?;
    let mut server = lock(&handle);
    let id = server.create_surface(w, h)?;
    server.map_surface_to_output(id, 0, 0);
    if let Err(e) = flush(shared, &mut server) {
        tracing::warn!(error = %e, "EGFX surface PDUs not sent");
        return None;
    }
    Some(id)
}

/// Send ResetGraphics for w×h and delete all surfaces; the caller then calls `create_surface`.
pub fn resize(shared: &GfxShared, w: u16, h: u16) {
    let Some(handle) = current_handle(shared) else {
        return;
    };
    let mut server = lock(&handle);
    server.resize(w, h);
    if let Err(e) = flush(shared, &mut server) {
        tracing::warn!(error = %e, "EGFX resize PDUs not sent");
    }
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

    #[test]
    fn cannot_submit_before_ready() {
        let shared = Arc::new(GfxShared::default());
        let factory = MacGfxFactory::new(Arc::clone(&shared));
        let (_bridge, _handle) = factory.build_server_with_handle().unwrap();
        assert!(matches!(
            can_submit(&shared, 0, false),
            Err(FrameSubmissionError::NotReady)
        ));
    }

    #[test]
    fn cannot_submit_without_connection() {
        let shared = GfxShared::default();
        assert!(matches!(
            can_submit(&shared, 0, false),
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
