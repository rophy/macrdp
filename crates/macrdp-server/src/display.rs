use bytes::Bytes;
use ironrdp_displaycontrol::pdu::DisplayControlMonitorLayout;
use ironrdp_egfx::server::FrameSubmissionError;
use ironrdp_pdu::pointer::PointerPositionAttribute;
use ironrdp_server::{
    BitmapUpdate, DesktopSize, DisplayUpdate, PixelFormat as RdpPixelFormat, RdpServerDisplay,
    RdpServerDisplayUpdates, ServerError, ServerErrorExt as _, ServerResult,
};
use macrdp_audio::SharedAudioTx;
use macrdp_capture::{
    get_cursor_position, CaptureConfig, CapturePixelFormat, CapturedFrame, CgFallbackCapturer,
    FrameData, ScreenCapturer,
};
use macrdp_encode::{self, Avc444EncodedFrame, EncodedFrame, Quality, VideoEncoder};
use std::num::{NonZeroU16, NonZeroUsize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::gfx::{self, SharedGfx};
use crate::handler::MouseCoordMapper;
use crate::perf_stats::SharedPerfStats;

/// Dirty area below this fraction of total pixels = "low activity" (0.5%).
const LOW_ACTIVITY_FRACTION: f64 = 0.005;
/// After this many consecutive low-activity frames, start reducing FPS.
const LOW_ACTIVITY_RAMP_FRAMES: u32 = 60;
/// Minimum FPS during sustained low activity.
const LOW_ACTIVITY_MIN_FPS: u32 = 5;
/// First wait before a failed BGRA capture restart is tried again; doubles up to the maximum.
const BGRA_RESTART_BACKOFF_MIN: std::time::Duration = std::time::Duration::from_secs(1);
const BGRA_RESTART_BACKOFF_MAX: std::time::Duration = std::time::Duration::from_secs(30);

/// Progressive rendering: on scene changes (large dirty area), temporarily reduce
/// encoder bitrate for fast initial delivery, then ramp quality back over several frames.
#[derive(Debug)]
struct ProgressiveRamp {
    trigger_fraction: f64,
    ramp_frames: u32,
    initial_quality: f32,
    remaining: u32,
}

impl ProgressiveRamp {
    fn new() -> Self {
        Self {
            trigger_fraction: 0.30,
            ramp_frames: 5,
            initial_quality: 0.25,
            remaining: 0,
        }
    }

    /// Evaluate a frame and return the bitrate action to take.
    fn evaluate(
        &mut self,
        dirty_pixels: u64,
        total_pixels: u64,
        base_bitrate: u32,
    ) -> ProgressiveAction {
        if total_pixels == 0 {
            return ProgressiveAction::None;
        }

        let dirty_frac = dirty_pixels as f64 / total_pixels as f64;

        if dirty_frac >= self.trigger_fraction && self.remaining == 0 {
            self.remaining = self.ramp_frames;
            let reduced = (base_bitrate as f32 * self.initial_quality) as u32;
            return ProgressiveAction::SceneChange {
                bitrate: reduced.max(500_000),
                dirty_pct: dirty_frac * 100.0,
            };
        }

        if self.remaining > 0 {
            self.remaining -= 1;
            if self.remaining == 0 {
                return ProgressiveAction::RampComplete {
                    bitrate: base_bitrate,
                };
            }
            let progress = 1.0 - (self.remaining as f32 / self.ramp_frames as f32);
            let factor = self.initial_quality + (1.0 - self.initial_quality) * progress;
            let ramped = (base_bitrate as f32 * factor) as u32;
            return ProgressiveAction::Ramp {
                bitrate: ramped.max(500_000),
            };
        }

        ProgressiveAction::None
    }

    #[cfg(test)]
    fn is_ramping(&self) -> bool {
        self.remaining > 0
    }
}

#[derive(Debug, PartialEq)]
enum ProgressiveAction {
    None,
    SceneChange { bitrate: u32, dirty_pct: f64 },
    Ramp { bitrate: u32 },
    RampComplete { bitrate: u32 },
}

/// Decide what to do with a captured frame before encoding.
#[derive(Debug, PartialEq)]
enum FramePlan {
    /// The pipeline cannot take a frame now: drop it without encoding, so the
    /// encoder state still matches what the client has.
    Skip,
    Encode {
        force_keyframe: bool,
    },
}

fn plan_frame(can_submit: Result<(), FrameSubmissionError>, last_send_failed: bool) -> FramePlan {
    match can_submit {
        Ok(()) => FramePlan::Encode {
            // The client missed the previous frame: its references are stale.
            force_keyframe: last_send_failed,
        },
        Err(_) => FramePlan::Skip,
    }
}

/// Encoder bitrate for a measured EGFX frame round-trip time (send to FrameAcknowledge).
///
/// Same steps as the old fork's `GfxState` network quality: full rate below 15 ms,
/// 80 % below 30 ms, 50 % below 60 ms, 30 % below 100 ms, else 15 %.
/// No measurement yet: full rate.
fn adaptive_bitrate(base: u32, rtt_ms: Option<f32>) -> u32 {
    let Some(rtt) = rtt_ms else {
        return base;
    };
    let quality = if rtt < 15.0 {
        1.0
    } else if rtt < 30.0 {
        0.8
    } else if rtt < 60.0 {
        0.5
    } else if rtt < 100.0 {
        0.3
    } else {
        0.15
    };
    (f64::from(base) * quality).round() as u32
}

/// The client must keep asking for no output this long before output stops.
/// Some clients pulse SuppressOutput under load (see ironrdp-server
/// `display_suppressed_handle`).
const SUPPRESS_DEBOUNCE: std::time::Duration = std::time::Duration::from_secs(1);
/// How often the loop checks the suppress flag while output is stopped.
const SUPPRESS_POLL: std::time::Duration = std::time::Duration::from_millis(100);

#[derive(Debug, PartialEq)]
enum SuppressState {
    /// Send output.
    Off,
    /// The client does not show output: do not encode or send frames.
    On,
    /// Output starts again after a stop: the next frame must be a keyframe.
    Resumed,
}

/// Debounced view of the client's SuppressOutput flag.
///
/// It engages only after a frame reached the client on this connection (mstsc
/// sends SuppressOutput during its connect handshake) and after the flag stays
/// set for `SUPPRESS_DEBOUNCE`. It clears at once.
#[derive(Debug, Default)]
struct SuppressGate {
    /// When the flag was first seen set (after the first delivered frame)
    since: Option<Instant>,
    active: bool,
}

impl SuppressGate {
    fn poll(&mut self, flag: bool, delivered: bool, now: Instant) -> SuppressState {
        if !flag {
            self.since = None;
            if self.active {
                self.active = false;
                return SuppressState::Resumed;
            }
            return SuppressState::Off;
        }
        if self.active {
            return SuppressState::On;
        }
        if !delivered {
            return SuppressState::Off;
        }
        let since = *self.since.get_or_insert(now);
        if now.duration_since(since) >= SUPPRESS_DEBOUNCE {
            self.active = true;
            SuppressState::On
        } else {
            SuppressState::Off
        }
    }
}

/// The last H.264 keyframe that reached a client. Sent first on the next
/// connection, so the client shows the screen before the encoder makes a new one.
#[derive(Clone)]
struct CachedKeyframe {
    /// AVC420 stream, or the AVC444 luma (main view) stream
    main: Bytes,
    /// AVC444 chroma (aux view) stream; None for AVC420
    chroma: Option<Bytes>,
    width: u16,
    height: u16,
    /// EGFX generation (channel) the keyframe was sent on
    generation: u64,
}

type KeyframeSlot = Arc<Mutex<Option<CachedKeyframe>>>;

/// What woke up the update loop.
enum LoopEvent {
    CursorTick,
    /// Time to try the held frame again.
    Retry,
    Frame(Option<CapturedFrame>),
}

/// Result of handling one captured frame.
enum FrameOutcome {
    /// The server must send this update.
    Update(DisplayUpdate),
    /// Sent through EGFX, or dropped for good.
    Done,
    /// Not sent yet (EGFX not ready, backpressure, overload): keep it and try again,
    /// so the last screen content reaches the client even when the screen stops changing.
    Hold(CapturedFrame),
}

/// One encoded frame, ready to submit.
enum Encoded {
    Avc420(EncodedFrame),
    Avc444(Avc444EncodedFrame),
}

impl Encoded {
    fn len(&self) -> usize {
        match self {
            Encoded::Avc420(f) => f.data.len(),
            Encoded::Avc444(f) => f.main_view.data.len() + f.aux_view.data.len(),
        }
    }

    fn is_keyframe(&self) -> bool {
        match self {
            Encoded::Avc420(f) => f.is_keyframe,
            Encoded::Avc444(f) => f.main_view.is_keyframe,
        }
    }
}

type EncoderSlot = Arc<Mutex<Option<Box<dyn VideoEncoder>>>>;

/// Display adapter that bridges ScreenCapturer to ironrdp-server
pub struct MacDisplay {
    width: u16,
    height: u16,
    /// Maximum capture resolution (physical display resolution).
    /// Client can resize freely up to this limit.
    max_width: u16,
    max_height: u16,
    /// Shared mouse coordinate mapper — updated on resize
    coord_mapper: MouseCoordMapper,
    /// Whether resolution is fixed by config (true) or follows client (false)
    fixed_resolution: bool,
    frame_rate: u32,
    quality: Quality,
    encoder_pref: macrdp_encode::EncoderPreference,
    /// Whether AVC444 mode is requested by config
    mode_444: bool,
    show_cursor: bool,
    base_bitrate: u32,
    /// New desktop size from a Display Control layout, sent as a Resize update
    pending_resize: PendingResize,
    /// Shared audio sender slot — updated per connection by AudioFactory
    shared_audio_tx: Option<SharedAudioTx>,
    /// Shared performance statistics collector (None = disabled)
    perf_stats: Option<SharedPerfStats>,
    /// EGFX pipeline of the current connection
    gfx: SharedGfx,
    /// Shared encoder slot — returned here when a connection ends, reused on reconnect
    encoder_cache: EncoderSlot,
    /// Resolution the cached encoder was created for
    cached_encoder_dims: (u16, u16),
    /// Last keyframe that reached a client, replayed on the next connection
    last_keyframe: KeyframeSlot,
    /// Set by the server while the client sends SuppressOutput (e.g. minimised)
    display_suppressed: Arc<AtomicBool>,
}

/// Desktop size from a Display Control layout that the client has not been sent yet
/// (sent as a Resize update). Shared by the display, its update loop and the
/// connection handler.
#[derive(Clone, Default)]
pub struct PendingResize(Arc<Mutex<Option<(u16, u16)>>>);

impl PendingResize {
    pub(crate) fn set(&self, width: u16, height: u16) {
        *self.0.lock().unwrap() = Some((width, height));
    }

    pub(crate) fn take(&self) -> Option<(u16, u16)> {
        self.0.lock().unwrap().take()
    }

    /// Drop a resize left over from an earlier connection: a new connection starts
    /// at the current size. Call once per connection, before its display starts.
    pub fn clear(&self) {
        if let Some((w, h)) = self.take() {
            tracing::debug!(w, h, "Dropping resize of the previous connection");
        }
    }
}

impl MacDisplay {
    #[allow(clippy::too_many_arguments)] // reason: plain constructor of the display state
    pub fn new(
        width: u16,
        height: u16,
        fixed_resolution: bool,
        frame_rate: u32,
        quality: Quality,
        encoder_pref: macrdp_encode::EncoderPreference,
        mode_444: bool,
        show_cursor: bool,
        bitrate_override: Option<u32>,
        coord_mapper: MouseCoordMapper,
        shared_audio_tx: Option<SharedAudioTx>,
        perf_stats: Option<SharedPerfStats>,
        gfx: SharedGfx,
        display_suppressed: Arc<AtomicBool>,
    ) -> Self {
        let base_bitrate = bitrate_override.unwrap_or_else(|| {
            macrdp_encode::screen_bitrate(width as u32, height as u32, frame_rate as f32, quality)
        });
        tracing::info!(
            base_bitrate_mbps = base_bitrate as f64 / 1_000_000.0,
            "Base bitrate"
        );
        // Max resolution: use physical display size so clients can resize freely.
        // SCK scales_to_fit handles any resolution.
        let (max_w, max_h) = macrdp_capture::detect_display_scale()
            .map(|scale| {
                (
                    width.max((width as u32 * scale) as u16),
                    height.max((height as u32 * scale) as u16),
                )
            })
            .unwrap_or((width.max(3840), height.max(2160)));
        Self {
            width,
            height,
            max_width: max_w,
            max_height: max_h,
            coord_mapper,
            fixed_resolution,
            frame_rate,
            quality,
            encoder_pref,
            mode_444,
            show_cursor,
            base_bitrate,
            pending_resize: PendingResize::default(),
            shared_audio_tx,
            perf_stats,
            gfx,
            encoder_cache: Arc::new(Mutex::new(None)),
            cached_encoder_dims: (0, 0),
            last_keyframe: Arc::new(Mutex::new(None)),
            display_suppressed,
        }
    }

    /// The resize slot, for clearing it when a new connection starts.
    pub fn pending_resize(&self) -> PendingResize {
        self.pending_resize.clone()
    }

    /// The largest desktop a client may ask for.
    pub fn max_size(&self) -> DesktopSize {
        DesktopSize {
            width: self.max_width,
            height: self.max_height,
        }
    }

    /// Adopt a client-requested desktop size, clamped to the maximum.
    fn resize_to(&mut self, width: u16, height: u16) {
        if self.fixed_resolution {
            tracing::debug!("Ignoring resize request — resolution is fixed by config");
            return;
        }
        let w = width.min(self.max_width);
        let h = height.min(self.max_height);
        if w > 0 && h > 0 && (w != self.width || h != self.height) {
            tracing::info!(
                old_w = self.width,
                old_h = self.height,
                new_w = w,
                new_h = h,
                "Adopting client-requested resolution"
            );
            self.width = w;
            self.height = h;
            self.base_bitrate = macrdp_encode::screen_bitrate(
                w as u32,
                h as u32,
                self.frame_rate as f32,
                self.quality,
            );
            self.coord_mapper.update_rdp_size(w, h);
        }
    }

    /// Reuse the encoder of the previous connection if the size is the same, else create one.
    fn take_or_create_encoder(&mut self) -> Option<Box<dyn VideoEncoder>> {
        if self.cached_encoder_dims == (self.width, self.height) {
            if let Some(mut enc) = self.encoder_cache.lock().unwrap().take() {
                tracing::info!("Reusing cached H.264 encoder from previous connection");
                enc.force_keyframe();
                // The adaptive bitrate of the previous connection does not apply.
                enc.set_bitrate(self.base_bitrate);
                return Some(enc);
            }
        }
        // A cached encoder for another size is of no use.
        self.encoder_cache.lock().unwrap().take();
        self.cached_encoder_dims = (self.width, self.height);
        match macrdp_encode::create_encoder(
            self.width as u32,
            self.height as u32,
            self.frame_rate as f32,
            self.quality,
            self.encoder_pref,
            self.mode_444,
            self.base_bitrate,
        ) {
            Ok(enc) => Some(enc),
            Err(e) => {
                tracing::warn!("No H.264 encoder ({e:#}) — video uses bitmap updates");
                None
            }
        }
    }
}

#[async_trait::async_trait]
impl RdpServerDisplay for MacDisplay {
    async fn size(&mut self) -> DesktopSize {
        DesktopSize {
            width: self.width,
            height: self.height,
        }
    }

    async fn request_initial_size(&mut self, client_size: DesktopSize) -> DesktopSize {
        self.resize_to(client_size.width, client_size.height);
        self.size().await
    }

    fn request_layout(&mut self, layout: DisplayControlMonitorLayout) {
        if self.fixed_resolution {
            tracing::debug!("Ignoring DisplayControl layout — resolution is fixed by config");
            return;
        }
        let primary = layout.monitors().iter().find(|m| m.is_primary());
        let Some(monitor) = primary else {
            tracing::warn!("DisplayControl layout has no primary monitor");
            return;
        };
        let (w, h) = monitor.dimensions();
        let (w, h) = (w as u16, h as u16);
        if w == self.width && h == self.height {
            return;
        }
        self.resize_to(w, h);
        self.pending_resize.set(self.width, self.height);
    }

    async fn updates(&mut self) -> ServerResult<Box<dyn RdpServerDisplayUpdates>> {
        // Do not clear pending_resize here: within a connection, a layout can arrive
        // after the previous loop returned Resize and before this call (the server
        // handles layouts on another thread). It is cleared once per connection
        // (PendingResize::clear from the connection handler).
        let encoder = self.take_or_create_encoder();
        if encoder.is_some() {
            tracing::info!("H.264 encoder available — will use GFX path when client supports it");
        }
        let capture_config = CaptureConfig {
            width: self.width as u32,
            height: self.height as u32,
            frame_rate: self.frame_rate,
            // The hardware AVC420 encoder takes NV12 pixel buffers without a copy.
            // Everything else (software encoder, AVC444, bitmap updates) needs BGRA.
            pixel_format: if encoder.is_some()
                && self.encoder_pref == macrdp_encode::EncoderPreference::Hardware
                && !self.mode_444
            {
                CapturePixelFormat::Nv12
            } else {
                CapturePixelFormat::Bgra
            },
            show_cursor: self.show_cursor,
        };
        // Read the current audio sender from the shared slot (set by AudioFactory per connection)
        let audio_tx = self
            .shared_audio_tx
            .as_ref()
            .and_then(|shared| shared.lock().unwrap().clone());
        let capturer = ScreenCapturer::new(capture_config.clone(), audio_tx)
            .await
            .map_err(|e| ServerError::reason("screen capture", format!("{e:#}")))?;

        let cursor_interval = if !self.show_cursor {
            let mut interval = tokio::time::interval(std::time::Duration::from_millis(16));
            interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);
            Some(interval)
        } else {
            None
        };

        Ok(Box::new(MacDisplayUpdates {
            capturer,
            capture_config,
            shared_audio_tx: self.shared_audio_tx.clone(),
            pending_resize: self.pending_resize.clone(),
            low_activity_frames: 0,
            current_fps: self.frame_rate,
            cursor_interval,
            last_cursor_pos: (0, 0),
            cursor_initialized: false,
            encoder,
            encoder_cache: Arc::clone(&self.encoder_cache),
            gfx: Arc::clone(&self.gfx),
            surface_frames: 0,
            held_frame: None,
            retry_at: tokio::time::Instant::now(),
            last_send_failed: false,
            avc_checked: None,
            need_bgra_capture: false,
            full_bitmap_next: true,
            mode_444: self.mode_444,
            base_bitrate: self.base_bitrate,
            last_applied_bitrate: self.base_bitrate,
            last_keyframe: Arc::clone(&self.last_keyframe),
            replay_pending: true,
            display_suppressed: Arc::clone(&self.display_suppressed),
            suppress: SuppressGate::default(),
            delivered: false,
            bgra_restart_after: None,
            bgra_restart_backoff: BGRA_RESTART_BACKOFF_MIN,
            progressive: ProgressiveRamp::new(),
            display_frame_count: 0,
            skip_next_frame: false,
            overload_count: 0,
            perf_stats: self.perf_stats.clone(),
            start: Instant::now(),
        }))
    }
}

struct MacDisplayUpdates {
    capturer: ScreenCapturer,
    capture_config: CaptureConfig,
    /// Audio sender slot, to restart capture with audio
    shared_audio_tx: Option<SharedAudioTx>,
    /// New desktop size from a Display Control layout, sent as a Resize update
    pending_resize: PendingResize,
    /// Consecutive frames with tiny dirty area (activity-based FPS reduction).
    low_activity_frames: u32,
    /// Current capture FPS (to avoid redundant set_frame_rate calls).
    current_fps: u32,
    /// Separate cursor channel: poll interval for reading cursor position
    cursor_interval: Option<tokio::time::Interval>,
    /// Last cursor position sent via PointerPosition PDU
    last_cursor_pos: (u16, u16),
    /// Whether the initial DefaultPointer has been sent
    cursor_initialized: bool,
    encoder: Option<Box<dyn VideoEncoder>>,
    /// Shared slot to return the encoder on drop for reuse across connections
    encoder_cache: EncoderSlot,
    gfx: SharedGfx,
    /// Newest frame not sent yet; tried again after one frame interval
    held_frame: Option<CapturedFrame>,
    /// When to try the held frame again
    retry_at: tokio::time::Instant,
    /// H.264 frames sent on the current surface
    surface_frames: u64,
    /// The last encoded frame did not reach the client: next frame must be a keyframe.
    last_send_failed: bool,
    /// (EGFX generation, AVC420 allowed) of the last capability check
    avc_checked: Option<(u64, bool)>,
    /// Bitmap updates need BGRA, but capture gives NV12: restart capture before the next frame.
    need_bgra_capture: bool,
    /// Send the whole frame as the next bitmap (screen was not painted by bitmaps before).
    full_bitmap_next: bool,
    mode_444: bool,
    /// Encoder bitrate at full network quality
    base_bitrate: u32,
    /// Last bitrate applied to the encoder (to avoid redundant set_bitrate calls)
    last_applied_bitrate: u32,
    /// Last keyframe that reached a client (shared across connections)
    last_keyframe: KeyframeSlot,
    /// Send the cached keyframe on the first surface of this connection
    replay_pending: bool,
    /// Client SuppressOutput flag, set by the server
    display_suppressed: Arc<AtomicBool>,
    suppress: SuppressGate,
    /// A frame (H.264 or bitmap) reached the client on this connection
    delivered: bool,
    /// A BGRA capture restart failed: do not try again before this time
    bgra_restart_after: Option<tokio::time::Instant>,
    /// Wait after the next failed BGRA capture restart
    bgra_restart_backoff: std::time::Duration,
    progressive: ProgressiveRamp,
    display_frame_count: u64,
    /// Encode overload protection: drop the next frame
    skip_next_frame: bool,
    /// Counter for rate-limiting encode overload warnings
    overload_count: u64,
    /// Shared performance statistics collector (None = disabled)
    perf_stats: Option<SharedPerfStats>,
    /// Time base for EGFX frame timestamps
    start: Instant,
}

impl Drop for MacDisplayUpdates {
    fn drop(&mut self) {
        if let Some(enc) = self.encoder.take() {
            *self.encoder_cache.lock().unwrap() = Some(enc);
        }
    }
}

#[async_trait::async_trait]
impl RdpServerDisplayUpdates for MacDisplayUpdates {
    /// Returns pointer, resize and bitmap updates. H.264 frames go through the
    /// EGFX pipeline inside this loop; a frame that is sent that way or dropped
    /// does not end the call (returning `Ok(None)` disconnects the client).
    async fn next_update(&mut self) -> ServerResult<Option<DisplayUpdate>> {
        loop {
            if let Some((w, h)) = self.pending_resize.take() {
                return Ok(Some(DisplayUpdate::Resize(DesktopSize {
                    width: w,
                    height: h,
                })));
            }

            // Send initial DefaultPointer when cursor channel is active
            if self.cursor_interval.is_some() && !self.cursor_initialized {
                self.cursor_initialized = true;
                let (x, y) = get_cursor_position();
                self.last_cursor_pos = (x, y);
                tracing::info!("Cursor channel active — sending initial pointer at ({x}, {y})");
                return Ok(Some(DisplayUpdate::DefaultPointer));
            }

            if self.need_bgra_capture
                && self
                    .bgra_restart_after
                    .is_none_or(|t| tokio::time::Instant::now() >= t)
            {
                self.restart_capture_bgra().await;
            }

            let suppressed = self.poll_suppress();

            let mut cursor_iv = self.cursor_interval.take();
            let has_cursor = cursor_iv.is_some();
            // While output is stopped, wake up now and then to see if it can start again.
            let has_timer = self.held_frame.is_some() || suppressed;
            let wake_at = if suppressed {
                tokio::time::Instant::now() + SUPPRESS_POLL
            } else {
                self.retry_at
            };
            let event = tokio::select! {
                biased;
                _ = async { cursor_iv.as_mut().expect("checked").tick().await }, if has_cursor => {
                    LoopEvent::CursorTick
                }
                _ = tokio::time::sleep_until(wake_at), if has_timer => LoopEvent::Retry,
                frame = self.next_frame() => LoopEvent::Frame(frame),
            };
            self.cursor_interval = cursor_iv;

            let frame = match event {
                LoopEvent::CursorTick => {
                    let (x, y) = get_cursor_position();
                    if (x, y) != self.last_cursor_pos {
                        self.last_cursor_pos = (x, y);
                        return Ok(Some(DisplayUpdate::PointerPosition(
                            PointerPositionAttribute { x, y },
                        )));
                    }
                    continue;
                }
                LoopEvent::Retry if suppressed => continue,
                LoopEvent::Retry => match self.held_frame.take() {
                    Some(frame) => frame,
                    None => continue,
                },
                LoopEvent::Frame(None) => return Ok(None),
                LoopEvent::Frame(Some(frame)) => {
                    // A newer frame replaces a held one: H.264 frames are full frames.
                    self.held_frame = None;
                    self.adjust_fps_for_activity(&frame);
                    // Check again: the flag may have changed while we waited.
                    if self.poll_suppress() {
                        // Keep only the newest frame; it is sent when output starts again.
                        self.held_frame = Some(frame);
                        continue;
                    }
                    frame
                }
            };

            match self.handle_frame(frame) {
                FrameOutcome::Update(update) => {
                    if matches!(update, DisplayUpdate::Bitmap(_)) {
                        self.delivered = true;
                    }
                    return Ok(Some(update));
                }
                FrameOutcome::Done => {}
                FrameOutcome::Hold(frame) => {
                    // A fixed deadline: the cursor tick must not keep pushing it back.
                    let interval_ms = 1000 / u64::from(self.capture_config.frame_rate.max(1));
                    self.retry_at =
                        tokio::time::Instant::now() + std::time::Duration::from_millis(interval_ms);
                    self.held_frame = Some(frame);
                }
            }
        }
    }
}

impl MacDisplayUpdates {
    /// Whether output is stopped because the client does not show it.
    /// When output starts again, the next frame is a full frame / keyframe.
    fn poll_suppress(&mut self) -> bool {
        let flag = self.display_suppressed.load(Ordering::Relaxed);
        let was_active = self.suppress.active;
        match self.suppress.poll(flag, self.delivered, Instant::now()) {
            SuppressState::Off => false,
            SuppressState::On => {
                if !was_active {
                    tracing::info!(
                        "Client does not show output (SuppressOutput) — encoding stopped"
                    );
                }
                true
            }
            SuppressState::Resumed => {
                tracing::info!("Client shows output again — sending a keyframe");
                if let Some(encoder) = &mut self.encoder {
                    encoder.force_keyframe();
                }
                self.full_bitmap_next = true;
                // Send the newest frame now.
                self.retry_at = tokio::time::Instant::now();
                false
            }
        }
    }

    /// Wait for the next video frame, handling SCK recovery and frame draining.
    async fn next_frame(&mut self) -> Option<CapturedFrame> {
        loop {
            let event = match self.capturer.next_frame().await {
                Some(e) => e,
                None => {
                    tracing::warn!(
                        "SCStream stopped — switching to CoreGraphics fallback (lock screen?)"
                    );
                    let fallback = CgFallbackCapturer::new(&self.capture_config);
                    loop {
                        match ScreenCapturer::new(self.capture_config.clone(), None).await {
                            Ok(new_capturer) => {
                                tracing::info!(
                                    "SCStream recovered — switching back from CoreGraphics"
                                );
                                self.capturer = new_capturer;
                                break;
                            }
                            Err(_) => {
                                if let Some(cg_frame) = fallback.capture_frame() {
                                    return Some(cg_frame);
                                }
                                tokio::time::sleep(fallback.frame_interval()).await;
                            }
                        }
                    }
                    continue;
                }
            };
            let frame = match event {
                macrdp_capture::CaptureEvent::Frame(f) => f,
                macrdp_capture::CaptureEvent::Idle => continue,
            };
            // Frames that wait already are newer: send only the newest one.
            let capturer = &mut self.capturer;
            return Some(newest_frame(frame, || capturer.try_next_frame()));
        }
    }

    /// Restart capture with BGRA pixels (bitmap updates cannot use NV12 buffers).
    async fn restart_capture_bgra(&mut self) {
        self.need_bgra_capture = false;
        self.capture_config.pixel_format = CapturePixelFormat::Bgra;
        let audio_tx = self
            .shared_audio_tx
            .as_ref()
            .and_then(|shared| shared.lock().unwrap().clone());
        match ScreenCapturer::new(self.capture_config.clone(), audio_tx).await {
            Ok(capturer) => {
                tracing::info!("Capture restarted with BGRA pixels for bitmap updates");
                self.capturer = capturer;
                self.bgra_restart_after = None;
                self.bgra_restart_backoff = BGRA_RESTART_BACKOFF_MIN;
            }
            Err(e) => {
                let wait = self.bgra_restart_backoff;
                tracing::warn!(
                    retry_in_s = wait.as_secs(),
                    "Capture restart with BGRA pixels failed: {e:#}"
                );
                self.bgra_restart_after = Some(tokio::time::Instant::now() + wait);
                self.bgra_restart_backoff = (wait * 2).min(BGRA_RESTART_BACKOFF_MAX);
            }
        }
    }

    /// Reduce capture FPS when dirty area is consistently tiny, restore instantly on activity.
    fn adjust_fps_for_activity(&mut self, frame: &CapturedFrame) {
        let total_pixels = frame.width as u64 * frame.height as u64;
        if total_pixels == 0 || frame.dirty_rects.is_empty() {
            return;
        }

        let dirty_pixels: u64 = frame
            .dirty_rects
            .iter()
            .map(|r| r.width as u64 * r.height as u64)
            .sum();
        let dirty_frac = dirty_pixels as f64 / total_pixels as f64;

        if dirty_frac < LOW_ACTIVITY_FRACTION {
            self.low_activity_frames = self.low_activity_frames.saturating_add(1);

            if self.low_activity_frames == LOW_ACTIVITY_RAMP_FRAMES
                && self.current_fps != LOW_ACTIVITY_MIN_FPS
            {
                tracing::info!(
                    from_fps = self.current_fps,
                    to_fps = LOW_ACTIVITY_MIN_FPS,
                    "Low screen activity — reducing capture FPS"
                );
                let _ = self.capturer.set_frame_rate(LOW_ACTIVITY_MIN_FPS);
                self.current_fps = LOW_ACTIVITY_MIN_FPS;
            }
        } else if self.low_activity_frames >= LOW_ACTIVITY_RAMP_FRAMES {
            let target_fps = self.capture_config.frame_rate;
            tracing::info!(
                from_fps = self.current_fps,
                to_fps = target_fps,
                "Screen activity resumed — restoring capture FPS"
            );
            let _ = self.capturer.set_frame_rate(target_fps);
            self.current_fps = target_fps;
            self.low_activity_frames = 0;
        } else {
            self.low_activity_frames = 0;
        }
    }

    fn apply_progressive_bitrate(&mut self, frame: &CapturedFrame) {
        let encoder = match &mut self.encoder {
            Some(e) => e,
            None => return,
        };

        let total_pixels = frame.width as u64 * frame.height as u64;
        let dirty_pixels: u64 = frame
            .dirty_rects
            .iter()
            .map(|r| r.width as u64 * r.height as u64)
            .sum();

        match self
            .progressive
            .evaluate(dirty_pixels, total_pixels, self.last_applied_bitrate)
        {
            ProgressiveAction::SceneChange { bitrate, dirty_pct } => {
                encoder.set_bitrate(bitrate);
                encoder.force_keyframe();
                tracing::info!(
                    dirty_pct = format!("{:.0}", dirty_pct),
                    reduced_mbps = bitrate as f64 / 1_000_000.0,
                    "Progressive: scene change detected, starting quality ramp"
                );
            }
            ProgressiveAction::Ramp { bitrate } => {
                encoder.set_bitrate(bitrate);
            }
            ProgressiveAction::RampComplete { bitrate } => {
                encoder.set_bitrate(bitrate);
                tracing::debug!("Progressive: ramp complete, restored full bitrate");
            }
            ProgressiveAction::None => {}
        }
    }

    /// Send one captured frame: as a bitmap update for the server to send, or as
    /// H.264 through EGFX.
    fn handle_frame(&mut self, frame: CapturedFrame) -> FrameOutcome {
        // Encode overload protection: wait one frame interval if previous encode took too long
        if self.skip_next_frame {
            self.skip_next_frame = false;
            return FrameOutcome::Hold(frame);
        }

        if self.encoder.is_none() {
            return self.bitmap_outcome(&frame);
        }

        if !self.gfx.ready.load(Ordering::SeqCst) {
            // H.264 encoder exists — never send bitmaps, wait for GFX to become ready.
            // Mixing bitmap and GFX causes 0xd06 DECOMPRESSION_FAILED on reconnect.
            return FrameOutcome::Hold(frame);
        }
        let generation = self.gfx.generation.load(Ordering::SeqCst);

        // A client can open EGFX with AVC disabled: then use bitmaps on this connection.
        let avc_allowed = match self.avc_checked {
            Some((checked, allowed)) if checked == generation => allowed,
            _ => {
                let Some(allowed) = gfx::avc420_allowed(&self.gfx) else {
                    // This connection's channel is not open yet.
                    return FrameOutcome::Hold(frame);
                };
                if !allowed {
                    tracing::info!("EGFX client has AVC disabled — using bitmap updates");
                    self.full_bitmap_next = true;
                }
                self.avc_checked = Some((generation, allowed));
                allowed
            }
        };
        if !avc_allowed {
            return self.bitmap_outcome(&frame);
        }

        if self.send_h264(&frame, generation) {
            FrameOutcome::Done
        } else {
            FrameOutcome::Hold(frame)
        }
    }

    fn bitmap_outcome(&mut self, frame: &CapturedFrame) -> FrameOutcome {
        match self.bitmap_update(frame) {
            Some(update) => FrameOutcome::Update(update),
            None => FrameOutcome::Done,
        }
    }

    /// Encode a frame as full-frame H.264 and submit it to the EGFX pipeline.
    /// Returns false if the frame did not reach the pipeline (the caller keeps it).
    fn send_h264(&mut self, frame: &CapturedFrame, generation: u64) -> bool {
        let (Ok(w), Ok(h)) = (u16::try_from(frame.width), u16::try_from(frame.height)) else {
            return true;
        };
        if w == 0 || h == 0 {
            return true;
        }

        // A new channel (generation) or a new frame size needs a new surface. The
        // surface state is shared: after a resize this is a new display loop on
        // the same channel, which still has the old surface.
        let Some((surface_id, created)) = gfx::ensure_surface(&self.gfx, generation, w, h) else {
            return false;
        };
        if created {
            tracing::info!(
                surface_id,
                width = w,
                height = h,
                generation,
                "EGFX surface created"
            );
            self.surface_frames = 0;
            if let Some(encoder) = &mut self.encoder {
                encoder.force_keyframe();
            }
            if std::mem::take(&mut self.replay_pending) {
                self.replay_keyframe(surface_id, w, h, generation);
            }
        }

        let use_444 = self.mode_444
            && self.gfx.avc444.load(Ordering::SeqCst)
            && self.encoder.as_ref().is_some_and(|e| e.supports_444());

        let state = gfx::can_submit(&self.gfx, surface_id, use_444);
        if let Err(FrameSubmissionError::UnknownSurface { .. }) = state {
            // The pipeline dropped our surface: create a new one with the next frame.
            gfx::forget_surface(&self.gfx);
        }
        match plan_frame(state, self.last_send_failed) {
            FramePlan::Skip => return false,
            FramePlan::Encode { force_keyframe } => {
                if force_keyframe {
                    if let Some(encoder) = &mut self.encoder {
                        encoder.force_keyframe();
                    }
                }
            }
        }

        self.apply_adaptive_bitrate();
        // Progressive rendering: detect scene change and apply quality ramp
        self.apply_progressive_bitrate(frame);

        self.display_frame_count += 1;
        let t0 = Instant::now();
        let Some(encoded) = self.encode(frame, use_444) else {
            // The encoder may have moved on without output: keep the frame and
            // send it again as a keyframe.
            self.last_send_failed = true;
            return false;
        };
        let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let frame_bytes = encoded.len();
        let is_keyframe = encoded.is_keyframe();

        let ts_ms = self.start.elapsed().as_millis() as u32;
        let result = match &encoded {
            Encoded::Avc420(f) => gfx::submit_avc420(&self.gfx, surface_id, &f.data, w, h, ts_ms),
            Encoded::Avc444(f) => gfx::submit_avc444(
                &self.gfx,
                surface_id,
                &f.main_view.data,
                &f.aux_view.data,
                w,
                h,
                ts_ms,
            ),
        };
        let sent = result.is_ok();
        match result {
            Ok(()) => {
                self.surface_frames += 1;
                if self.surface_frames == 1 || self.surface_frames.is_multiple_of(600) {
                    tracing::info!(
                        surface_frames = self.surface_frames,
                        is_keyframe,
                        avc444 = matches!(encoded, Encoded::Avc444(_)),
                        rtt_ms = ?gfx::rtt_ms(&self.gfx),
                        "EGFX H.264 frames sent"
                    );
                }
                self.last_send_failed = false;
                self.delivered = true;
                self.cache_keyframe(&encoded, w, h, generation);
                tracing::debug!(
                    display_frame = self.display_frame_count,
                    h264_bytes = frame_bytes,
                    is_keyframe,
                    encode_ms = format!("{:.1}", encode_ms),
                    "Display: sent GFX frame"
                );
            }
            Err(e) => {
                // The encoder moved on but the client did not get this frame.
                self.last_send_failed = true;
                tracing::warn!(
                    display_frame = self.display_frame_count,
                    error = %e,
                    "EGFX frame not sent — next frame is a keyframe"
                );
            }
        }

        if let Some(ps) = &self.perf_stats {
            ps.lock()
                .unwrap()
                .record_frame(encode_ms, frame_bytes as u32, is_keyframe);
        }
        let frame_interval_ms = 1000.0 / self.capture_config.frame_rate as f64;
        if encode_ms > frame_interval_ms * 0.95 {
            self.skip_next_frame = true;
            self.overload_count += 1;
            // Rate-limit log: warn once per ~60 overloads
            if self.overload_count % 60 == 1 {
                tracing::warn!(
                    encode_ms = format!("{:.1}", encode_ms),
                    frame_interval_ms = format!("{:.1}", frame_interval_ms),
                    overload_total = self.overload_count,
                    "encode overload — skipping frames"
                );
            }
        }
        // Keep a frame that did not reach the client: it is sent again as a keyframe.
        sent
    }

    /// Follow the EGFX frame round-trip time, with 10 % hysteresis.
    fn apply_adaptive_bitrate(&mut self) {
        let Some(encoder) = &mut self.encoder else {
            return;
        };
        let target = adaptive_bitrate(self.base_bitrate, gfx::rtt_ms(&self.gfx));
        let change = (f64::from(target) - f64::from(self.last_applied_bitrate)).abs();
        if target != self.last_applied_bitrate && change > f64::from(self.base_bitrate) * 0.1 {
            tracing::info!(
                from_mbps = self.last_applied_bitrate as f64 / 1_000_000.0,
                to_mbps = target as f64 / 1_000_000.0,
                rtt_ms = ?gfx::rtt_ms(&self.gfx),
                "Adaptive bitrate adjustment"
            );
            encoder.set_bitrate(target);
            self.last_applied_bitrate = target;
        }
    }

    /// Keep a keyframe that reached the client, for the next connection.
    fn cache_keyframe(&self, encoded: &Encoded, width: u16, height: u16, generation: u64) {
        let cached = match encoded {
            Encoded::Avc420(f) if f.is_keyframe => CachedKeyframe {
                main: f.data.clone(),
                chroma: None,
                width,
                height,
                generation,
            },
            // Both streams must be IDR, else the chroma stream needs older frames.
            Encoded::Avc444(f) if f.main_view.is_keyframe && f.aux_view.is_keyframe => {
                CachedKeyframe {
                    main: f.main_view.data.clone(),
                    chroma: Some(f.aux_view.data.clone()),
                    width,
                    height,
                    generation,
                }
            }
            _ => return,
        };
        *self.last_keyframe.lock().unwrap() = Some(cached);
    }

    /// Send the keyframe of the previous connection on a new surface of the same size.
    /// The client then shows the screen at once; the encoder's next frame is a keyframe too.
    fn replay_keyframe(&mut self, surface_id: u16, w: u16, h: u16, generation: u64) {
        let Some(cached) = self.last_keyframe.lock().unwrap().clone() else {
            return;
        };
        if cached.generation == generation {
            // Same channel (e.g. a resize in this session): the client has newer content.
            return;
        }
        if (cached.width, cached.height) != (w, h) {
            tracing::debug!(
                cached_w = cached.width,
                cached_h = cached.height,
                w,
                h,
                "Cached keyframe has another size — not replayed"
            );
            return;
        }
        let avc444 = cached.chroma.is_some();
        if avc444 && !self.gfx.avc444.load(Ordering::SeqCst) {
            tracing::debug!(
                "Cached keyframe is AVC444 but the client has no AVC444 — not replayed"
            );
            return;
        }
        if let Err(e) = gfx::can_submit(&self.gfx, surface_id, avc444) {
            tracing::debug!(error = %e, "Cached keyframe not replayed");
            return;
        }
        let ts_ms = self.start.elapsed().as_millis() as u32;
        let result = match &cached.chroma {
            None => gfx::submit_avc420(&self.gfx, surface_id, &cached.main, w, h, ts_ms),
            Some(chroma) => {
                gfx::submit_avc444(&self.gfx, surface_id, &cached.main, chroma, w, h, ts_ms)
            }
        };
        match result {
            Ok(()) => tracing::info!(
                width = w,
                height = h,
                avc444,
                "Replaying cached keyframe for instant reconnect image"
            ),
            Err(e) => tracing::warn!(error = %e, "Cached keyframe replay failed"),
        }
    }

    /// Encode a full frame: NV12 pixel buffer (zero copy), BGRA AVC444, or BGRA AVC420.
    fn encode(&mut self, frame: &CapturedFrame, use_444: bool) -> Option<Encoded> {
        let display_frame = self.display_frame_count;
        let encoder = self.encoder.as_mut()?;
        let bgra = match &frame.data {
            FrameData::PixelBuffer(buf) => {
                // Zero-copy VT path — encode CVPixelBuffer directly
                return match encoder.encode_pixel_buffer(buf.as_ptr(), false) {
                    Ok(encoded) if !encoded.data.is_empty() => Some(Encoded::Avc420(encoded)),
                    Ok(_) => {
                        tracing::warn!(display_frame, "Zero-copy encode returned empty data");
                        None
                    }
                    Err(e) => {
                        tracing::warn!(display_frame, "Zero-copy encode failed: {e}");
                        None
                    }
                };
            }
            FrameData::Raw(bytes) => bytes,
        };

        // AVC444 dual-stream path
        if use_444 {
            match encoder.encode_bgra_444(bgra, frame.width, frame.height, frame.stride) {
                Ok(encoded) if !encoded.main_view.data.is_empty() => {
                    return Some(Encoded::Avc444(encoded));
                }
                Ok(_) => {
                    tracing::warn!(
                        display_frame,
                        "AVC444 encode returned EMPTY data — frame dropped!"
                    );
                    return None;
                }
                Err(e) => {
                    tracing::warn!(
                        display_frame,
                        "AVC444 encode failed: {e}, falling back to AVC420"
                    );
                    // Fall through to AVC420 path below
                }
            }
        }

        // AVC420 path (default or fallback from AVC444 failure)
        match encoder.encode_bgra(bgra, frame.width, frame.height, frame.stride) {
            Ok(encoded) if !encoded.data.is_empty() => Some(Encoded::Avc420(encoded)),
            Ok(_) => {
                tracing::warn!(
                    display_frame,
                    "H.264 encode returned EMPTY data — frame dropped!"
                );
                None
            }
            Err(e) => {
                tracing::warn!(display_frame, "H.264 encode failed: {e}");
                None
            }
        }
    }

    /// Bitmap update for the dirty area of a frame (whole frame when needed).
    fn bitmap_update(&mut self, frame: &CapturedFrame) -> Option<DisplayUpdate> {
        // Bitmap path: requires BGRA raw bytes
        let bgra_bitmap = match &frame.data {
            FrameData::Raw(bytes) => bytes,
            FrameData::PixelBuffer(_) => {
                self.need_bgra_capture = true;
                self.full_bitmap_next = true;
                return None;
            }
        };

        if !self.full_bitmap_next && !frame.dirty_rects.is_empty() {
            // Find bounding box of all dirty rects to send a single update
            let mut min_x = frame.width;
            let mut min_y = frame.height;
            let mut max_x = 0u32;
            let mut max_y = 0u32;

            for r in &frame.dirty_rects {
                min_x = min_x.min(r.x);
                min_y = min_y.min(r.y);
                max_x = max_x.max(r.x + r.width);
                max_y = max_y.max(r.y + r.height);
            }

            // Clamp to frame bounds
            max_x = max_x.min(frame.width);
            max_y = max_y.min(frame.height);

            if max_x > min_x && max_y > min_y {
                let w = max_x - min_x;
                let h = max_y - min_y;
                let width = NonZeroU16::new(w as u16)?;
                let height = NonZeroU16::new(h as u16)?;

                // Extract only the dirty region from the full frame buffer
                let bpp = 4usize;
                let dirty_stride = w as usize * bpp;
                let mut dirty_data = Vec::with_capacity(dirty_stride * h as usize);
                for row in min_y..max_y {
                    let src_offset = row as usize * frame.stride + min_x as usize * bpp;
                    let src_end = src_offset + dirty_stride;
                    if src_end <= bgra_bitmap.len() {
                        dirty_data.extend_from_slice(&bgra_bitmap[src_offset..src_end]);
                    }
                }

                let stride = NonZeroUsize::new(dirty_stride)?;

                return Some(DisplayUpdate::Bitmap(BitmapUpdate {
                    x: min_x as u16,
                    y: min_y as u16,
                    width,
                    height,
                    format: RdpPixelFormat::BgrA32,
                    data: Bytes::from(dirty_data),
                    stride,
                }));
            }
        }

        // No dirty rects available — send full frame (first frame or fallback)
        let width = NonZeroU16::new(frame.width as u16)?;
        let height = NonZeroU16::new(frame.height as u16)?;
        let stride = NonZeroUsize::new(frame.stride)?;
        self.full_bitmap_next = false;

        Some(DisplayUpdate::Bitmap(BitmapUpdate {
            x: 0,
            y: 0,
            width,
            height,
            format: RdpPixelFormat::BgrA32,
            data: bgra_bitmap.clone(),
            stride,
        }))
    }
}

/// Coalesce a frame with the frames that already wait behind it: the newest frame wins,
/// with the dirty rects of all of them (bitmap updates send only the dirty area).
/// `Idle` events in between are skipped.
fn newest_frame(
    mut frame: CapturedFrame,
    mut next: impl FnMut() -> Option<macrdp_capture::CaptureEvent>,
) -> CapturedFrame {
    while let Some(event) = next() {
        if let macrdp_capture::CaptureEvent::Frame(mut newer) = event {
            // Empty dirty rects mean "whole frame changed"; keep that if either says so.
            if frame.dirty_rects.is_empty() {
                newer.dirty_rects.clear();
            } else if !newer.dirty_rects.is_empty() {
                newer.dirty_rects.append(&mut frame.dirty_rects);
            }
            frame = newer;
        }
    }
    frame
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_frame(timestamp_us: u64, rects: &[(u32, u32)]) -> CapturedFrame {
        CapturedFrame {
            width: 64,
            height: 64,
            data: FrameData::Raw(Bytes::new()),
            stride: 256,
            timestamp_us,
            dirty_rects: rects
                .iter()
                .map(|&(x, y)| macrdp_capture::Rect {
                    x,
                    y,
                    width: 1,
                    height: 1,
                })
                .collect(),
        }
    }

    fn drain(first: CapturedFrame, waiting: Vec<macrdp_capture::CaptureEvent>) -> CapturedFrame {
        let mut waiting = waiting.into_iter();
        newest_frame(first, || waiting.next())
    }

    #[test]
    fn newest_frame_keeps_last_change() {
        use macrdp_capture::CaptureEvent::{Frame, Idle};

        // Nothing waits: the frame itself.
        assert_eq!(drain(test_frame(1, &[(0, 0)]), vec![]).timestamp_us, 1);

        // Newer frames wait (with Idle in between): the newest is sent, never none.
        let got = drain(
            test_frame(1, &[(1, 1)]),
            vec![
                Frame(test_frame(2, &[(2, 2)])),
                Idle,
                Frame(test_frame(3, &[(3, 3)])),
            ],
        );
        assert_eq!(got.timestamp_us, 3);
        let mut xs: Vec<u32> = got.dirty_rects.iter().map(|r| r.x).collect();
        xs.sort_unstable();
        assert_eq!(xs, [1, 2, 3], "dirty areas of skipped frames are kept");

        // A frame with no dirty rects (whole frame) makes the result whole too.
        let got = drain(test_frame(1, &[]), vec![Frame(test_frame(2, &[(2, 2)]))]);
        assert!(got.dirty_rects.is_empty());
        let got = drain(test_frame(1, &[(1, 1)]), vec![Frame(test_frame(2, &[]))]);
        assert!(got.dirty_rects.is_empty());

        // Only Idle waits: the frame itself.
        assert_eq!(drain(test_frame(4, &[]), vec![Idle]).timestamp_us, 4);
    }

    #[test]
    fn frame_gate() {
        assert_eq!(
            plan_frame(
                Err(FrameSubmissionError::Backpressured {
                    frames_in_flight: 3,
                    max_frames_in_flight: 3
                }),
                false
            ),
            FramePlan::Skip
        );
        assert_eq!(
            plan_frame(Err(FrameSubmissionError::NotReady), false),
            FramePlan::Skip
        );
        assert_eq!(
            plan_frame(Ok(()), false),
            FramePlan::Encode {
                force_keyframe: false
            }
        );
        assert_eq!(
            plan_frame(Ok(()), true),
            FramePlan::Encode {
                force_keyframe: true
            }
        );
    }

    #[test]
    fn adaptive_bitrate_policy() {
        // Thresholds of the old fork's GfxState::adaptive_bitrate.
        assert_eq!(adaptive_bitrate(10_000_000, None), 10_000_000);
        assert_eq!(adaptive_bitrate(10_000_000, Some(10.0)), 10_000_000);
        assert_eq!(adaptive_bitrate(10_000_000, Some(15.0)), 8_000_000);
        assert_eq!(adaptive_bitrate(10_000_000, Some(20.0)), 8_000_000);
        assert_eq!(adaptive_bitrate(10_000_000, Some(45.0)), 5_000_000);
        assert_eq!(adaptive_bitrate(10_000_000, Some(80.0)), 3_000_000);
        assert_eq!(adaptive_bitrate(10_000_000, Some(300.0)), 1_500_000);
        assert_eq!(adaptive_bitrate(10_000_000, Some(1000.0)), 1_500_000);
    }

    #[test]
    fn suppress_waits_for_first_delivered_frame() {
        let mut gate = SuppressGate::default();
        let t0 = Instant::now();
        assert_eq!(gate.poll(true, false, t0), SuppressState::Off);
        assert_eq!(
            gate.poll(true, false, t0 + 5 * SUPPRESS_DEBOUNCE),
            SuppressState::Off
        );
        // The debounce starts when a frame was delivered.
        let t1 = t0 + 6 * SUPPRESS_DEBOUNCE;
        assert_eq!(gate.poll(true, true, t1), SuppressState::Off);
        assert_eq!(
            gate.poll(true, true, t1 + SUPPRESS_DEBOUNCE),
            SuppressState::On
        );
    }

    #[test]
    fn suppress_debounce_and_resume() {
        let mut gate = SuppressGate::default();
        let t0 = Instant::now();
        let half = SUPPRESS_DEBOUNCE / 2;
        assert_eq!(gate.poll(true, true, t0), SuppressState::Off);
        assert_eq!(gate.poll(true, true, t0 + half), SuppressState::Off);
        // A short pulse does not stop output and is not a resume.
        assert_eq!(gate.poll(false, true, t0 + half), SuppressState::Off);
        let t1 = t0 + SUPPRESS_DEBOUNCE;
        assert_eq!(gate.poll(true, true, t1), SuppressState::Off);
        assert_eq!(gate.poll(true, true, t1 + half), SuppressState::Off);
        assert_eq!(
            gate.poll(true, true, t1 + SUPPRESS_DEBOUNCE),
            SuppressState::On
        );
        assert_eq!(
            gate.poll(true, true, t1 + 10 * SUPPRESS_DEBOUNCE),
            SuppressState::On
        );
        // Clears at once, and reports the resume once.
        let t2 = t1 + 11 * SUPPRESS_DEBOUNCE;
        assert_eq!(gate.poll(false, true, t2), SuppressState::Resumed);
        assert_eq!(gate.poll(false, true, t2), SuppressState::Off);
    }

    #[test]
    fn test_progressive_no_trigger_below_threshold() {
        let mut ramp = ProgressiveRamp::new();
        // 20% dirty — below 30% threshold
        let action = ramp.evaluate(200, 1000, 8_000_000);
        assert_eq!(action, ProgressiveAction::None);
        assert!(!ramp.is_ramping());
    }

    #[test]
    fn test_progressive_triggers_on_scene_change() {
        let mut ramp = ProgressiveRamp::new();
        // 50% dirty — above threshold
        let action = ramp.evaluate(500, 1000, 8_000_000);
        match action {
            ProgressiveAction::SceneChange { bitrate, .. } => {
                // 8M * 0.25 = 2M
                assert_eq!(bitrate, 2_000_000);
            }
            other => panic!("Expected SceneChange, got {:?}", other),
        }
        assert!(ramp.is_ramping());
    }

    #[test]
    fn test_progressive_ramp_sequence() {
        let mut ramp = ProgressiveRamp::new();
        let base = 10_000_000u32;

        // Trigger scene change
        let action = ramp.evaluate(1000, 1000, base);
        assert!(matches!(action, ProgressiveAction::SceneChange { .. }));

        // Ramp frames 1..4 should produce increasing bitrates
        let mut prev_bitrate = 0u32;
        for i in 0..4 {
            let action = ramp.evaluate(0, 1000, base);
            match action {
                ProgressiveAction::Ramp { bitrate } => {
                    assert!(
                        bitrate > prev_bitrate,
                        "frame {i}: {bitrate} should be > {prev_bitrate}"
                    );
                    assert!(
                        bitrate < base,
                        "frame {i}: {bitrate} should be < base {base}"
                    );
                    prev_bitrate = bitrate;
                }
                other => panic!("frame {i}: expected Ramp, got {:?}", other),
            }
        }

        // Frame 5 should complete the ramp and restore base bitrate
        let action = ramp.evaluate(0, 1000, base);
        assert_eq!(action, ProgressiveAction::RampComplete { bitrate: base });
        assert!(!ramp.is_ramping());
    }

    #[test]
    fn test_progressive_no_retrigger_during_ramp() {
        let mut ramp = ProgressiveRamp::new();
        // Trigger
        ramp.evaluate(1000, 1000, 8_000_000);
        // Another scene change during ramp should NOT re-trigger
        let action = ramp.evaluate(1000, 1000, 8_000_000);
        assert!(matches!(action, ProgressiveAction::Ramp { .. }));
    }

    #[test]
    fn test_progressive_retrigger_after_ramp_complete() {
        let mut ramp = ProgressiveRamp::new();
        let base = 8_000_000;
        // Complete a full ramp
        ramp.evaluate(1000, 1000, base);
        for _ in 0..5 {
            ramp.evaluate(0, 1000, base);
        }
        assert!(!ramp.is_ramping());

        // Should be able to trigger again
        let action = ramp.evaluate(1000, 1000, base);
        assert!(matches!(action, ProgressiveAction::SceneChange { .. }));
    }

    #[test]
    fn test_progressive_minimum_bitrate_floor() {
        let mut ramp = ProgressiveRamp::new();
        // Very low base bitrate: 1 Mbps * 0.25 = 250 Kbps, should clamp to 500 Kbps
        let action = ramp.evaluate(1000, 1000, 1_000_000);
        match action {
            ProgressiveAction::SceneChange { bitrate, .. } => {
                assert_eq!(bitrate, 500_000);
            }
            other => panic!("Expected SceneChange, got {:?}", other),
        }
    }

    #[test]
    fn test_progressive_zero_total_pixels() {
        let mut ramp = ProgressiveRamp::new();
        let action = ramp.evaluate(0, 0, 8_000_000);
        assert_eq!(action, ProgressiveAction::None);
    }

    #[test]
    fn test_progressive_exact_threshold() {
        let mut ramp = ProgressiveRamp::new();
        // Exactly 30% should trigger
        let action = ramp.evaluate(300, 1000, 8_000_000);
        assert!(matches!(action, ProgressiveAction::SceneChange { .. }));
    }

    #[test]
    fn test_progressive_just_below_threshold() {
        let mut ramp = ProgressiveRamp::new();
        // 29.9% should not trigger
        let action = ramp.evaluate(299, 1000, 8_000_000);
        assert_eq!(action, ProgressiveAction::None);
    }
}
