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
    cursor, get_cursor_position, CaptureConfig, CapturePixelFormat, CapturedFrame,
    CgFallbackCapturer, FrameData, SafePixelBuffer, ScreenCapturer,
};
use macrdp_encode::{self, Avc444EncodedFrame, EncodedFrame, Quality, VideoEncoder};
use std::num::{NonZeroU16, NonZeroUsize};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::gfx::{self, SharedGfx};
use crate::handler::MouseCoordMapper;
use crate::perf_stats::SharedPerfStats;
use crate::pointer::PointerCache;
use crate::refine::{self, TileGrid, TileRect};

/// Dirty area below this fraction of total pixels = "low activity" (0.5%).
const LOW_ACTIVITY_FRACTION: f64 = 0.005;
/// After this many consecutive low-activity frames, start reducing FPS.
const LOW_ACTIVITY_RAMP_FRAMES: u32 = 60;
/// Minimum FPS during sustained low activity.
const LOW_ACTIVITY_MIN_FPS: u32 = 5;
/// First wait before a failed BGRA capture restart is tried again; doubles up to the maximum.
const BGRA_RESTART_BACKOFF_MIN: std::time::Duration = std::time::Duration::from_secs(1);
const BGRA_RESTART_BACKOFF_MAX: std::time::Duration = std::time::Duration::from_secs(30);

/// Lossless refinement starts when no H.264 frame was sent for this long.
const REFINE_DELAY: std::time::Duration = std::time::Duration::from_millis(200);
/// Lossless refinement is sent in frames of about this size.
const REFINE_CHUNK_BYTES: usize = 64 * 1024;
/// How often to check if the client acknowledged the last refinement frame.
const REFINE_POLL: std::time::Duration = std::time::Duration::from_millis(4);

/// A scene change needs a quiet screen before it: no frame for this long.
const PROGRESSIVE_REARM_GAP: std::time::Duration = std::time::Duration::from_millis(250);

/// Progressive rendering: on scene changes (large dirty area), temporarily reduce
/// encoder bitrate for fast initial delivery, then ramp quality back over several frames.
/// A scene change is a large change after a quiet screen (a small change, or no
/// frame for `PROGRESSIVE_REARM_GAP`). Sustained motion or scrolling is one scene
/// change at its start, not one per ramp.
#[derive(Debug)]
struct ProgressiveRamp {
    trigger_fraction: f64,
    ramp_frames: u32,
    initial_quality: f32,
    remaining: u32,
    /// The screen was quiet since the last scene change.
    armed: bool,
    last_frame: Option<Instant>,
}

impl ProgressiveRamp {
    fn new() -> Self {
        Self {
            trigger_fraction: 0.30,
            ramp_frames: 5,
            initial_quality: 0.25,
            remaining: 0,
            armed: true,
            last_frame: None,
        }
    }

    /// Evaluate a frame and return the bitrate action to take.
    fn evaluate(
        &mut self,
        dirty_pixels: u64,
        total_pixels: u64,
        base_bitrate: u32,
        now: Instant,
    ) -> ProgressiveAction {
        if total_pixels == 0 {
            return ProgressiveAction::None;
        }

        let dirty_frac = dirty_pixels as f64 / total_pixels as f64;
        let large = dirty_frac >= self.trigger_fraction;
        let after_gap = self
            .last_frame
            .is_some_and(|t| now.duration_since(t) >= PROGRESSIVE_REARM_GAP);
        self.last_frame = Some(now);
        if !large || after_gap {
            self.armed = true;
        }

        if large && self.armed && self.remaining == 0 {
            self.armed = false;
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

/// What happened to an encoded frame.
#[derive(Debug, Clone, Copy, PartialEq)]
enum FrameResult {
    /// The client got the frame.
    Sent,
    /// The encoder dropped the frame on purpose (rate control). The frame is not
    /// a reference, so the client's references still match the encoder.
    Dropped,
    /// The encoder state moved on but the client did not get the frame
    /// (encode error, or the frame was not sent).
    Lost,
}

/// Whether the next frame must be a keyframe, after `result`.
/// A drop keeps the earlier state: a keyframe that was due stays due. A
/// keyframe that was asked for (`requested`: new surface, resume, scene
/// change) and dropped by the encoder is due too: the encoder clears its
/// request when it takes the frame, and no periodic keyframe follows.
fn keyframe_due(was_due: bool, requested: bool, result: FrameResult) -> bool {
    match result {
        FrameResult::Sent => false,
        FrameResult::Dropped => was_due || requested,
        FrameResult::Lost => true,
    }
}

/// Encoder bitrate share of each level: full rate, then 80 %, 50 %, 30 %, 15 %.
const LEVEL_QUALITY: [f64; 5] = [1.0, 0.8, 0.5, 0.3, 0.15];
/// Step down when the median queueing delay of the last `DOWN_SPAN` is above
/// this: the link is full. A client that decodes slower than the capture rate
/// queues frames too (mstsc in a VM, ~33 fps: 40-50 ms with a 3-frame window),
/// but a lower bitrate does not help it (measured: same fps at 13 and 44 Mbps).
const QUEUE_DELAY_HIGH_MS: f32 = 100.0;
const DOWN_SPAN: std::time::Duration = std::time::Duration::from_millis(300);
/// Fewest samples for a step down.
const MIN_DOWN_SAMPLES: usize = 5;
/// No second step down within this time: the queue needs time to drain.
const STEP_DOWN_HOLD: std::time::Duration = std::time::Duration::from_millis(500);
/// Step up when the median queueing delay of the last second stayed below
/// this for the up dwell.
const QUEUE_DELAY_LOW_MS: f32 = 50.0;
/// Fewest samples (in one second) for a step up.
const MIN_UP_SAMPLES: usize = 10;
/// Up dwell. It doubles (up to the maximum) when a step up is followed by a
/// step down within `PROBE_FAILED`: the link has no room for the next level.
const UP_DWELL_MIN: std::time::Duration = std::time::Duration::from_secs(3);
const UP_DWELL_MAX: std::time::Duration = std::time::Duration::from_secs(48);
const PROBE_FAILED: std::time::Duration = std::time::Duration::from_secs(10);
/// The up dwell goes back to the minimum after this long without a step down.
const DWELL_RESET: std::time::Duration = std::time::Duration::from_secs(30);
/// A quiet screen gives no samples, so the level cannot follow the link. A
/// congestion measured that long ago is stale: step up one level for each
/// `QUIET_STEP_UP` of quiet (a failed probe at the next motion steps down
/// within `DOWN_SPAN`).
const QUIET_STEP_UP: std::time::Duration = std::time::Duration::from_secs(10);

/// Bitrate level that follows congestion: the queueing delay of stream frames
/// (send-to-acknowledge time above the smallest one of the channel, see
/// `FrameWindow::queue_delay`). Path delay alone (a far client) and the Wi-Fi
/// wake-up delay after a quiet screen do not change it.
#[derive(Debug)]
struct BitrateLevel {
    /// Index into `LEVEL_QUALITY`; 0 = best.
    level: usize,
    /// Since when the queueing delay is below `QUEUE_DELAY_LOW_MS` (continuously).
    below_since: Option<Instant>,
    last_down: Option<Instant>,
    last_up: Option<Instant>,
    up_dwell: std::time::Duration,
    /// Since when there are too few samples to measure (quiet screen)
    quiet_since: Option<Instant>,
    /// Last update with enough samples
    last_sample: Option<Instant>,
}

impl Default for BitrateLevel {
    fn default() -> Self {
        Self {
            level: 0,
            below_since: None,
            last_down: None,
            last_up: None,
            up_dwell: UP_DWELL_MIN,
            quiet_since: None,
            last_sample: None,
        }
    }
}

impl BitrateLevel {
    /// Feed the median queueing delay (ms) and sample count of the last
    /// `DOWN_SPAN` (`short`) and of the last second (`long`); returns the
    /// bitrate share (0..=1) to use. Too few samples (a quiet screen): no
    /// change, except one step up for each `QUIET_STEP_UP` of quiet.
    fn update(
        &mut self,
        short: Option<(f32, usize)>,
        long: Option<(f32, usize)>,
        now: Instant,
    ) -> f64 {
        if self
            .last_down
            .is_some_and(|t| now.duration_since(t) >= DWELL_RESET)
        {
            self.up_dwell = UP_DWELL_MIN;
        }
        let measured = long.filter(|&(_, n)| n >= MIN_UP_SAMPLES);
        let short_quiet = short.is_none_or(|(_, n)| n < MIN_DOWN_SAMPLES);
        if measured.is_none() && short_quiet {
            // No frames are sent on a quiet screen, so this is called again
            // only when the screen changes: count the time since the last samples.
            let since = *self
                .quiet_since
                .get_or_insert(self.last_sample.unwrap_or(now));
            let steps =
                (now.duration_since(since).as_secs_f64() / QUIET_STEP_UP.as_secs_f64()) as u32;
            if steps > 0 {
                self.quiet_since = Some(since + QUIET_STEP_UP * steps);
                if self.level > 0 {
                    self.level = self.level.saturating_sub(steps as usize);
                    self.last_up = Some(now);
                }
            }
        } else {
            self.quiet_since = None;
            self.last_sample = Some(now);
        }
        let congested =
            short.is_some_and(|(d, n)| n >= MIN_DOWN_SAMPLES && d > QUEUE_DELAY_HIGH_MS);
        if congested {
            self.below_since = None;
            let held = self
                .last_down
                .is_some_and(|t| now.duration_since(t) < STEP_DOWN_HOLD);
            if !held && self.level + 1 < LEVEL_QUALITY.len() {
                if self
                    .last_up
                    .is_some_and(|t| now.duration_since(t) < PROBE_FAILED)
                {
                    self.up_dwell = (self.up_dwell * 2).min(UP_DWELL_MAX);
                    self.last_up = None;
                }
                self.level += 1;
                self.last_down = Some(now);
            }
            return LEVEL_QUALITY[self.level];
        }
        match measured {
            Some((d, _)) if d < QUEUE_DELAY_LOW_MS && self.level > 0 => {
                let since = *self.below_since.get_or_insert(now);
                if now.duration_since(since) >= self.up_dwell {
                    self.level -= 1;
                    self.below_since = None;
                    self.last_up = Some(now);
                }
            }
            _ => self.below_since = None,
        }
        LEVEL_QUALITY[self.level]
    }
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
    /// The screen is quiet: send the changed text tiles losslessly.
    Refine,
    /// The refinement frame is compressed (None: compression failed)
    RefineReady(Option<gfx::CompressedPdus>),
    /// Time to try the held frame again.
    Retry,
    Frame(Option<CapturedFrame>),
}

/// A refinement frame that waits for its compression (see `gfx::ClearFrame`).
struct RefineJob {
    frame: gfx::ClearFrame,
    compressed: tokio::sync::oneshot::Receiver<Option<gfx::CompressedPdus>>,
    /// Tells the compression to stop: the frame went out as it is
    cancel: Arc<AtomicBool>,
    /// Pending tiles looked at, and of those sent lossless (for the log)
    tiles: (usize, usize),
    started: Instant,
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

/// A BGRA frame that the client got as H.264.
struct BgraFrame {
    data: BgraPixels,
    width: u32,
    height: u32,
    stride: usize,
}

/// Pixels of a `BgraFrame`: copied bytes, or the capture pixel buffer itself.
enum BgraPixels {
    Bytes(Bytes),
    Buffer(SafePixelBuffer),
}

impl BgraFrame {
    /// Call `f` with the BGRA bytes and the bytes per row.
    fn read<R>(&self, f: impl FnOnce(&[u8], usize) -> R) -> Option<R> {
        match &self.data {
            BgraPixels::Bytes(b) => Some(f(b, self.stride)),
            BgraPixels::Buffer(buf) => buf.read_bgra(f),
        }
    }
}

/// H.264 regions and lossless refinement (see `refine`).
#[derive(Default)]
struct RefineState {
    /// Tiles that changed since the last H.264 frame that reached the client
    h264_dirty: Option<TileGrid>,
    /// Tiles the client shows from H.264 since the last refinement
    pending: Option<TileGrid>,
    /// Last frame sent as H.264
    last_frame: Option<BgraFrame>,
    /// When to refine
    due: Option<tokio::time::Instant>,
}

/// What the next H.264 frame updates on the client (AVC420 regions).
#[derive(Debug, PartialEq, Eq)]
enum Regions {
    /// The whole frame
    Whole,
    /// These tiles
    Tiles(Vec<TileRect>),
    /// Nothing changed: no frame to send
    Nothing,
}

impl RefineState {
    /// Grid of the frame size; a new size starts with all tiles set.
    fn grid(slot: &mut Option<TileGrid>, width: u32, height: u32) -> &mut TileGrid {
        if slot.as_ref().is_none_or(|g| g.size() != (width, height)) {
            let mut g = TileGrid::new(width, height);
            g.mark_all();
            *slot = Some(g);
        }
        slot.as_mut().expect("set above")
    }

    /// A frame was captured: its dirty tiles go into the next H.264 frame.
    fn on_captured(&mut self, frame: &CapturedFrame) {
        Self::grid(&mut self.h264_dirty, frame.width, frame.height).mark_dirty(&frame.dirty_rects);
    }

    /// A new surface is blank: the next H.264 frame must cover all of it,
    /// also when that frame is not sent at the first try.
    fn on_new_surface(&mut self, width: u32, height: u32) {
        Self::grid(&mut self.h264_dirty, width, height).mark_all();
    }

    /// Regions for the next H.264 frame.
    fn regions(&mut self, width: u32, height: u32) -> Regions {
        let grid = Self::grid(&mut self.h264_dirty, width, height);
        let (cols, rows) = grid.cols_rows();
        match grid.tiles().count() {
            0 => Regions::Nothing,
            n if n == (cols * rows) as usize => Regions::Whole,
            _ => Regions::Tiles(grid.rects()),
        }
    }

    /// The H.264 frame reached the client: its tiles now need refinement.
    fn on_sent(&mut self, frame: &CapturedFrame) {
        let (w, h) = (frame.width, frame.height);
        let dirty = Self::grid(&mut self.h264_dirty, w, h).clone();
        self.h264_dirty.as_mut().expect("set").clear();
        let pending = Self::grid(&mut self.pending, w, h);
        pending.union(&dirty);
        let data = match &frame.data {
            FrameData::Raw(data) => BgraPixels::Bytes(data.clone()),
            FrameData::PixelBuffer(buf) if buf.is_bgra() => BgraPixels::Buffer(buf.clone_ref()),
            FrameData::PixelBuffer(_) => {
                self.last_frame = None;
                return;
            }
        };
        self.last_frame = Some(BgraFrame {
            data,
            width: w,
            height: h,
            stride: frame.stride,
        });
        self.due = Some(tokio::time::Instant::now() + REFINE_DELAY);
    }
}

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
    /// Lossless refinement of a quiet screen (AVC420 only)
    refine: bool,
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

    pub(crate) fn is_set(&self) -> bool {
        self.0.lock().unwrap().is_some()
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
        lossless_text: bool,
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
            refine: lossless_text,
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
        let (w, h) = clamp_size(
            u32::from(width),
            u32::from(height),
            (self.max_width, self.max_height),
        );
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
        tracing::debug!(
            monitors = ?layout
                .monitors()
                .iter()
                .map(|m| (m.dimensions(), m.is_primary()))
                .collect::<Vec<_>>(),
            "DisplayControl layout received"
        );
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
        // Compare after the clamp: at the maximum size, a larger layout (a
        // client window larger than the Mac display) changes nothing and must
        // not restart the display loop.
        let (w, h) = clamp_size(w, h, (self.max_width, self.max_height));
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
            // The hardware AVC420 encoder takes capture pixel buffers without a copy:
            // NV12, or BGRA when lossless refinement needs the exact pixels.
            // Everything else (software encoder, AVC444, bitmap updates) needs BGRA bytes.
            pixel_format: if encoder.is_some()
                && self.encoder_pref == macrdp_encode::EncoderPreference::Hardware
                && !self.mode_444
            {
                if self.refine {
                    CapturePixelFormat::BgraBuffer
                } else {
                    CapturePixelFormat::Nv12
                }
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
            pointer_cache: PointerCache::default(),
            cursor_seed: None,
            cursor_scale: 0.0,
            pending_pointer: None,
            coord_mapper: self.coord_mapper.clone(),
            encoder,
            encoder_cache: Arc::clone(&self.encoder_cache),
            gfx: Arc::clone(&self.gfx),
            surface_frames: 0,
            held_frame: None,
            resize_wait: None,
            bitmap_fallback: false,
            retry_at: tokio::time::Instant::now(),
            last_send_failed: false,
            // A new or reused encoder starts with a keyframe.
            keyframe_requested: true,
            avc_checked: None,
            need_bgra_capture: false,
            full_bitmap_next: true,
            mode_444: self.mode_444,
            refine: (self.refine && !self.mode_444).then(RefineState::default),
            refine_job: None,
            base_bitrate: self.base_bitrate,
            last_applied_bitrate: self.base_bitrate,
            bitrate_level: BitrateLevel::default(),
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
    /// Pointer shapes the client has in its cache
    pointer_cache: PointerCache,
    /// System cursor seed and picture scale of the last shape check
    cursor_seed: Option<i32>,
    cursor_scale: f64,
    /// Pointer update to send before anything else (position after a shape)
    pending_pointer: Option<DisplayUpdate>,
    /// Mouse mapping and the client's recent cursor moves
    coord_mapper: MouseCoordMapper,
    encoder: Option<Box<dyn VideoEncoder>>,
    /// Shared slot to return the encoder on drop for reuse across connections
    encoder_cache: EncoderSlot,
    gfx: SharedGfx,
    /// Newest frame not sent yet; tried again after one frame interval
    held_frame: Option<CapturedFrame>,
    /// A resize waits for the EGFX frames in flight since this time
    resize_wait: Option<tokio::time::Instant>,
    /// EGFX was closed by the client: frames go as bitmap updates
    bitmap_fallback: bool,
    /// When to try the held frame again
    retry_at: tokio::time::Instant,
    /// H.264 frames sent on the current surface
    surface_frames: u64,
    /// The last encoded frame did not reach the client: next frame must be a keyframe.
    last_send_failed: bool,
    /// A keyframe was asked for and not sent yet
    keyframe_requested: bool,
    /// (EGFX generation, AVC420 allowed) of the last capability check
    avc_checked: Option<(u64, bool)>,
    /// Bitmap updates need BGRA, but capture gives NV12: restart capture before the next frame.
    need_bgra_capture: bool,
    /// Send the whole frame as the next bitmap (screen was not painted by bitmaps before).
    full_bitmap_next: bool,
    mode_444: bool,
    /// Lossless refinement state; None when it is off
    refine: Option<RefineState>,
    /// Refinement frame being compressed
    refine_job: Option<RefineJob>,
    /// Encoder bitrate at full network quality
    base_bitrate: u32,
    /// Last bitrate applied to the encoder (to avoid redundant set_bitrate calls)
    last_applied_bitrate: u32,
    bitrate_level: BitrateLevel,
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
        // The pipeline counts the frame as sent: send it.
        self.flush_refine_job();
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
            if self.pending_resize.is_set() && self.wait_for_frames_before_resize() {
                tokio::time::sleep(RESIZE_DRAIN_POLL).await;
                continue;
            }
            if let Some((w, h)) = self.pending_resize.take() {
                // The client gets a new pointer cache with the new size: send the shape again.
                self.pointer_cache.clear();
                self.cursor_seed = None;
                return Ok(Some(DisplayUpdate::Resize(DesktopSize {
                    width: w,
                    height: h,
                })));
            }

            if let Some(update) = self.pending_pointer.take() {
                return Ok(Some(update));
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
            let refine_at = self.refine.as_ref().and_then(|r| r.due);
            let mut job = self.refine_job.take();
            let has_job = job.is_some();
            let event = tokio::select! {
                biased;
                done = async { (&mut job.as_mut().expect("checked").compressed).await }, if has_job => {
                    LoopEvent::RefineReady(done.ok().flatten())
                }
                _ = async { cursor_iv.as_mut().expect("checked").tick().await }, if has_cursor => {
                    LoopEvent::CursorTick
                }
                _ = tokio::time::sleep_until(wake_at), if has_timer => LoopEvent::Retry,
                frame = self.next_frame() => LoopEvent::Frame(frame),
                _ = async { tokio::time::sleep_until(refine_at.expect("checked")).await },
                    if refine_at.is_some() && !suppressed => LoopEvent::Refine,
            };
            self.cursor_interval = cursor_iv;
            self.refine_job = job;

            let frame = match event {
                LoopEvent::CursorTick => {
                    let shape = self.poll_cursor_shape();
                    let position = self.poll_cursor_position();
                    match (shape, position) {
                        (Some(shape), position) => {
                            self.pending_pointer = position;
                            return Ok(Some(shape));
                        }
                        (None, Some(position)) => return Ok(Some(position)),
                        (None, None) => continue,
                    }
                }
                LoopEvent::Refine => {
                    self.refine_quiet_screen();
                    continue;
                }
                LoopEvent::RefineReady(compressed) => {
                    if let Some(job) = self.refine_job.take() {
                        self.send_refine_job(job, compressed);
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
                    // Its dirty tiles stay marked for the next H.264 frame.
                    self.held_frame = None;
                    if let Some(r) = &mut self.refine {
                        r.on_captured(&frame);
                    }
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

/// A resize waits at most this long for the EGFX frames in flight ...
const RESIZE_DRAIN_MAX: std::time::Duration = std::time::Duration::from_millis(500);
/// ... and checks this often.
const RESIZE_DRAIN_POLL: std::time::Duration = std::time::Duration::from_millis(4);

impl MacDisplayUpdates {
    /// Before a resize, no EGFX frame may be in flight: the resize changes the
    /// client's desktop size at once, and FreeRDP cannot draw a frame of the
    /// old, larger surface into the smaller desktop (without cairo it fails
    /// the EndFrame and closes the graphics channel; the picture stops). So
    /// send no more frames and wait until the client acknowledged all of them
    /// (at most `RESIZE_DRAIN_MAX`). Returns true while it must wait.
    fn wait_for_frames_before_resize(&mut self) -> bool {
        self.flush_refine_job();
        let now = tokio::time::Instant::now();
        let since = *self.resize_wait.get_or_insert(now);
        let in_flight = gfx::frames_in_flight(&self.gfx);
        if in_flight > 0 && now.duration_since(since) < RESIZE_DRAIN_MAX {
            return true;
        }
        tracing::debug!(
            in_flight,
            waited_ms = now.duration_since(since).as_millis(),
            "Resize after the frames in flight"
        );
        self.resize_wait = None;
        false
    }

    /// Ask the encoder for a keyframe (see `keyframe_due`).
    fn request_keyframe(&mut self) {
        if let Some(encoder) = &mut self.encoder {
            encoder.force_keyframe();
        }
        self.keyframe_requested = true;
    }

    /// Capture interval; sets the EGFX frame window.
    fn frame_interval(&self) -> std::time::Duration {
        std::time::Duration::from_secs(1) / self.capture_config.frame_rate.max(1)
    }

    /// A PointerPosition update when the Mac cursor moved by itself.
    fn poll_cursor_position(&mut self) -> Option<DisplayUpdate> {
        let (x, y) = get_cursor_position();
        if (x, y) == self.last_cursor_pos {
            return None;
        }
        self.last_cursor_pos = (x, y);
        // Only moves the client did not cause: an echo makes a
        // scaling client move its cursor again (see is_client_move).
        if self.coord_mapper.is_client_move((x, y)) {
            return None;
        }
        // The client places its cursor in RDP desktop coordinates.
        let (x, y) = self.coord_mapper.to_rdp(x, y);
        // The client cursor moves to this position: older client
        // positions no longer describe where the client shows it.
        self.coord_mapper.forget_client_moves();
        Some(DisplayUpdate::PointerPosition(PointerPositionAttribute {
            x,
            y,
        }))
    }

    /// A pointer shape update when the Mac cursor image changed (I-beam, hand, hidden, ...).
    /// The cursor seed changes with each new image, so the image is read only then.
    fn poll_cursor_shape(&mut self) -> Option<DisplayUpdate> {
        let seed = cursor::cursor_seed();
        let scale = self.coord_mapper.picture_scale();
        if self.cursor_seed == Some(seed) && self.cursor_scale == scale {
            return None;
        }
        self.cursor_seed = Some(seed);
        self.cursor_scale = scale;
        let Some(shape) = cursor::current_cursor_shape(scale) else {
            tracing::debug!(seed, "No system cursor image");
            return None;
        };
        let update = self.pointer_cache.show(shape);
        if let Some(u) = &update {
            tracing::debug!(seed, update = ?u, "Cursor shape changed");
        }
        update
    }

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
                self.request_keyframe();
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

        match self.progressive.evaluate(
            dirty_pixels,
            total_pixels,
            self.last_applied_bitrate,
            Instant::now(),
        ) {
            ProgressiveAction::SceneChange { bitrate, dirty_pct } => {
                encoder.set_bitrate(bitrate);
                encoder.force_keyframe();
                self.keyframe_requested = true;
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

        if !self.gfx.ready.load(Ordering::SeqCst)
            && self.gfx.closed_by_client.load(Ordering::SeqCst)
        {
            // The client closed EGFX in this session (FreeRDP does so when it
            // cannot draw a frame). No EGFX frame can be sent any more, and the
            // channel is not opened again: send bitmaps, or the picture stops.
            if !self.bitmap_fallback {
                tracing::warn!("EGFX channel closed by the client: using bitmap updates");
                self.bitmap_fallback = true;
                self.full_bitmap_next = true;
            }
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
        // A refinement frame still being compressed goes first, as it is.
        self.flush_refine_job();
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
            self.request_keyframe();
            if let Some(r) = &mut self.refine {
                r.on_new_surface(frame.width, frame.height);
            }
            if std::mem::take(&mut self.replay_pending) {
                self.replay_keyframe(surface_id, w, h, generation);
            }
        }

        let regions = match &mut self.refine {
            Some(r) => r.regions(frame.width, frame.height),
            None => Regions::Whole,
        };
        let regions = match regions {
            Regions::Whole => Vec::new(),
            Regions::Tiles(tiles) => tiles,
            Regions::Nothing => {
                // The dirty rectangles hit no tile (all empty or outside).
                tracing::debug!("Display: no tile changed, frame not sent");
                return true;
            }
        };

        let use_444 = self.mode_444
            && self.gfx.avc444.load(Ordering::SeqCst)
            && self.encoder.as_ref().is_some_and(|e| e.supports_444());

        let state = gfx::can_submit(&self.gfx, surface_id, use_444, self.frame_interval());
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
        let encoded = match self.encode(frame, use_444) {
            Ok(encoded) => encoded,
            Err(result) => {
                // Keep the frame and try it again; a keyframe only when the
                // encoder may have moved on without output.
                self.last_send_failed =
                    keyframe_due(self.last_send_failed, self.keyframe_requested, result);
                return false;
            }
        };
        let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let frame_bytes = encoded.len();
        let is_keyframe = encoded.is_keyframe();

        let ts_ms = self.start.elapsed().as_millis() as u32;
        let result = match &encoded {
            Encoded::Avc420(f) => {
                gfx::submit_avc420(&self.gfx, surface_id, &f.data, &regions, w, h, ts_ms)
            }
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
                    let (window, base_rtt_ms, queue_depth) = gfx::window_stats(&self.gfx);
                    tracing::info!(
                        window,
                        base_rtt_ms,
                        queue_depth,
                        surface_frames = self.surface_frames,
                        is_keyframe,
                        avc444 = matches!(encoded, Encoded::Avc444(_)),
                        rtt_ms = ?gfx::rtt_ms(&self.gfx),
                        "EGFX H.264 frames sent"
                    );
                }
                self.last_send_failed = keyframe_due(
                    self.last_send_failed,
                    self.keyframe_requested,
                    FrameResult::Sent,
                );
                // The encoder took the request with this frame.
                self.keyframe_requested = false;
                self.delivered = true;
                if let Some(r) = &mut self.refine {
                    r.on_sent(frame);
                }
                self.cache_keyframe(&encoded, w, h, generation);
                tracing::debug!(
                    display_frame = self.display_frame_count,
                    h264_bytes = frame_bytes,
                    is_keyframe,
                    encode_ms = format!("{:.1}", encode_ms),
                    regions = regions.len(),
                    "Display: sent GFX frame"
                );
            }
            Err(e) => {
                // The encoder moved on but the client did not get this frame.
                self.last_send_failed = keyframe_due(
                    self.last_send_failed,
                    self.keyframe_requested,
                    FrameResult::Lost,
                );
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

    /// The screen is quiet: send the tiles that the client shows from H.264 and
    /// that have few colors (text, UI) again with ClearCodec (lossless).
    ///
    /// The tiles go in chunks of about `REFINE_CHUNK_BYTES`, one at a time
    /// (the next after the client acknowledged the last frame), so that a new
    /// H.264 frame waits behind one chunk at most.
    fn refine_quiet_screen(&mut self) {
        let Some(state) = &mut self.refine else {
            return;
        };
        state.due = None;
        if self.held_frame.is_some() {
            // A frame still waits: it comes first, then a new quiet period.
            return;
        }
        if self.refine_job.is_some() {
            // The last chunk is not sent yet.
            state.due = Some(tokio::time::Instant::now() + REFINE_POLL);
            return;
        }
        let Some(frame) = &state.last_frame else {
            return;
        };
        let Some(pending) = state.pending.as_mut().filter(|p| !p.is_empty()) else {
            return;
        };
        if pending.size() != (frame.width, frame.height) {
            return;
        }
        let Some(surface) = gfx::current_surface(&self.gfx) else {
            return;
        };
        if (u32::from(surface.width), u32::from(surface.height)) != (frame.width, frame.height) {
            return;
        }
        let interval = std::time::Duration::from_secs(1) / self.capture_config.frame_rate.max(1);
        if gfx::frames_in_flight(&self.gfx) > 0
            || gfx::can_submit(&self.gfx, surface.id, false, interval).is_err()
        {
            // Wait for the client to catch up.
            state.due = Some(tokio::time::Instant::now() + REFINE_POLL);
            return;
        }
        let t0 = Instant::now();
        let read = frame.read(|data, stride| take_refine_chunk(pending, data, stride));
        let Some(chunk) = read else {
            // The pixels cannot be read: give up on these tiles.
            pending.clear();
            return;
        };
        if !pending.is_empty() {
            state.due = Some(tokio::time::Instant::now());
        }
        if chunk.tiles.is_empty() {
            return;
        }
        let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let ts_ms = self.start.elapsed().as_millis() as u32;
        let frame = match gfx::take_clearcodec(&self.gfx, surface, chunk.tiles, ts_ms) {
            Ok(frame) => frame,
            Err(e) => {
                tracing::warn!(error = %e, "Refine: frame not sent");
                return;
            }
        };
        tracing::debug!(
            frame_id = frame.frame_id,
            bytes = frame.bytes,
            encode_ms = format!("{encode_ms:.1}"),
            "Refine: frame taken, compressing"
        );
        // Compression takes 10-30 ms: off the display loop, so that a new
        // frame does not wait for it.
        let (tx, rx) = tokio::sync::oneshot::channel();
        let cancel = Arc::new(AtomicBool::new(false));
        let (pdus, stop) = (frame.pdus(), Arc::clone(&cancel));
        tokio::task::spawn_blocking(move || {
            let _ = tx.send(gfx::compress_clearcodec(&pdus, &stop));
        });
        self.refine_job = Some(RefineJob {
            frame,
            compressed: rx,
            cancel,
            tiles: (chunk.total, chunk.picked),
            started: Instant::now(),
        });
    }

    /// Send the refinement frame now: compressed if that is done, else as it is.
    fn flush_refine_job(&mut self) {
        let Some(mut job) = self.refine_job.take() else {
            return;
        };
        let compressed = match job.compressed.try_recv() {
            Ok(compressed) => compressed,
            Err(_) => {
                job.cancel.store(true, Ordering::Relaxed);
                None
            }
        };
        self.send_refine_job(job, compressed);
    }

    fn send_refine_job(&mut self, job: RefineJob, compressed: Option<gfx::CompressedPdus>) {
        let was_compressed = compressed.is_some();
        match gfx::send_clearcodec(&self.gfx, &job.frame, compressed) {
            Ok(wire) => tracing::debug!(
                frame_id = job.frame.frame_id,
                tiles = job.tiles.0,
                lossless = job.tiles.1,
                bytes = job.frame.bytes,
                wire,
                compressed = was_compressed,
                wait_ms = format!("{:.1}", job.started.elapsed().as_secs_f64() * 1000.0),
                "Refine: sent lossless tiles"
            ),
            Err(e) => tracing::warn!(error = %e, "Refine: frame not sent"),
        }
    }

    /// Follow congestion (see `BitrateLevel`). The levels are at least 15 % of
    /// the base rate apart, so a level change always changes the bitrate.
    fn apply_adaptive_bitrate(&mut self) {
        let Some(encoder) = &mut self.encoder else {
            return;
        };
        let ms = |(d, n): (std::time::Duration, usize)| (d.as_secs_f32() * 1000.0, n);
        let short = gfx::queue_delay(&self.gfx, DOWN_SPAN).map(ms);
        let delay = gfx::queue_delay(&self.gfx, crate::frame_window::DELAY_SPAN).map(ms);
        let quality = self.bitrate_level.update(short, delay, Instant::now());
        let target = (f64::from(self.base_bitrate) * quality).round() as u32;
        if target != self.last_applied_bitrate {
            tracing::info!(
                from_mbps = self.last_applied_bitrate as f64 / 1_000_000.0,
                to_mbps = target as f64 / 1_000_000.0,
                queue_delay_ms = ?delay.map(|(d, _)| d),
                samples = ?delay.map(|(_, n)| n),
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
        if let Err(e) = gfx::can_submit(&self.gfx, surface_id, avc444, self.frame_interval()) {
            tracing::debug!(error = %e, "Cached keyframe not replayed");
            return;
        }
        let ts_ms = self.start.elapsed().as_millis() as u32;
        let result = match &cached.chroma {
            None => gfx::submit_avc420(&self.gfx, surface_id, &cached.main, &[], w, h, ts_ms),
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
    /// Without output, says if the encoder dropped the frame or lost it.
    fn encode(&mut self, frame: &CapturedFrame, use_444: bool) -> Result<Encoded, FrameResult> {
        let display_frame = self.display_frame_count;
        let encoder = self.encoder.as_mut().ok_or(FrameResult::Lost)?;
        let bgra = match &frame.data {
            FrameData::PixelBuffer(buf) => {
                // Zero-copy VT path — encode CVPixelBuffer directly
                return match encoder.encode_pixel_buffer(buf.as_ptr(), false) {
                    Ok(encoded) if !encoded.data.is_empty() => Ok(Encoded::Avc420(encoded)),
                    Ok(_) => {
                        tracing::debug!(display_frame, "Encoder dropped the frame");
                        Err(FrameResult::Dropped)
                    }
                    Err(e) => {
                        tracing::warn!(display_frame, "Zero-copy encode failed: {e}");
                        Err(FrameResult::Lost)
                    }
                };
            }
            FrameData::Raw(bytes) => bytes,
        };

        // AVC444 dual-stream path
        if use_444 {
            match encoder.encode_bgra_444(bgra, frame.width, frame.height, frame.stride) {
                Ok(encoded) if !encoded.main_view.data.is_empty() => {
                    return Ok(Encoded::Avc444(encoded));
                }
                Ok(_) => {
                    tracing::warn!(
                        display_frame,
                        "AVC444 encode returned EMPTY data — frame dropped!"
                    );
                    // The two views may not agree on what was dropped: start again
                    // from a keyframe.
                    return Err(FrameResult::Lost);
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
            Ok(encoded) if !encoded.data.is_empty() => Ok(Encoded::Avc420(encoded)),
            Ok(_) => {
                tracing::debug!(display_frame, "Encoder dropped the frame");
                Err(FrameResult::Dropped)
            }
            Err(e) => {
                tracing::warn!(display_frame, "H.264 encode failed: {e}");
                Err(FrameResult::Lost)
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

/// A requested desktop size, limited to `max`.
fn clamp_size(width: u32, height: u32, max: (u16, u16)) -> (u16, u16) {
    let clamp = |v: u32, m: u16| u16::try_from(v).unwrap_or(u16::MAX).min(m);
    (clamp(width, max.0), clamp(height, max.1))
}

/// One chunk of lossless refinement (see `take_refine_chunk`).
struct RefineChunk {
    /// ClearCodec bitmaps (sequence numbers are set when they are sent)
    tiles: Vec<(TileRect, Vec<u8>)>,
    /// Pending tiles looked at
    total: usize,
    /// Of those, tiles sent lossless
    picked: usize,
}

/// Take whole tile rows from `pending` until the ClearCodec data reaches
/// `REFINE_CHUNK_BYTES`. Text-like tiles of those rows are encoded; the other
/// tiles (pictures) stay H.264 and leave `pending` too.
fn take_refine_chunk(pending: &mut TileGrid, data: &[u8], stride: usize) -> RefineChunk {
    // A new encoder each time: its glyph cache starts empty, so it never
    // refers to a glyph the client may not have.
    let mut encoder = ironrdp_graphics::clearcodec::ClearCodecEncoder::new();
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
                    && refine::is_text_like(data, stride, pending.tile_rect(c, row))
            })
            .collect();
        for c in 0..cols {
            if pending.is_set(c, row) {
                pending.unset(c, row);
                chunk.total += 1;
                chunk.picked += usize::from(lossless[c as usize]);
            }
        }
        let runs = pending.runs_larger_than(refine::GLYPH_MAX_PIXELS, |c, r| {
            r == row && lossless[c as usize]
        });
        for rect in runs {
            let px = refine::copy_rect(data, stride, rect);
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

    /// A 128x64 BGRA frame (4x2 tiles of 32 px): `photo` tiles get a new color
    /// per pixel, the others two colors (text-like).
    fn refine_frame(dirty: &[(u32, u32, u32, u32)], photo: &[(u32, u32)]) -> CapturedFrame {
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
        CapturedFrame {
            width: w,
            height: h,
            data: FrameData::Raw(Bytes::from(data)),
            stride: (w * 4) as usize,
            timestamp_us: 0,
            dirty_rects: dirty
                .iter()
                .map(|&(x, y, width, height)| macrdp_capture::Rect {
                    x,
                    y,
                    width,
                    height,
                })
                .collect(),
        }
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
    fn refine_first_frame_covers_the_whole_surface() {
        let mut r = RefineState::default();
        let f = refine_frame(&[(40, 40, 2, 2)], &[]);
        r.on_captured(&f);
        assert_eq!(r.regions(128, 64), Regions::Whole);
    }

    #[test]
    fn refine_regions_are_the_changed_tiles() {
        let mut r = RefineState::default();
        let f = refine_frame(&[], &[]);
        r.on_captured(&f);
        r.on_sent(&f);
        let f = refine_frame(&[(40, 40, 2, 2)], &[]);
        r.on_captured(&f);
        assert_eq!(r.regions(128, 64), Regions::Tiles(vec![tile(1, 1, 1)]));
    }

    #[test]
    fn refine_no_changed_tile_is_not_the_whole_frame() {
        let mut r = RefineState::default();
        let f = refine_frame(&[], &[]);
        r.on_captured(&f);
        r.on_sent(&f);
        // Zero-size and outside rectangles touch no tile.
        let f = refine_frame(&[(10, 10, 0, 5), (500, 10, 4, 4)], &[]);
        r.on_captured(&f);
        assert_eq!(r.regions(128, 64), Regions::Nothing);
    }

    #[test]
    fn refine_new_surface_covers_all_until_sent() {
        let mut r = RefineState::default();
        let f = refine_frame(&[], &[]);
        r.on_captured(&f);
        r.on_sent(&f);
        r.on_new_surface(128, 64);
        // The first frame on the new surface was not sent (skipped or dropped):
        // the next one still covers the whole surface.
        let f = refine_frame(&[(40, 40, 2, 2)], &[]);
        r.on_captured(&f);
        assert_eq!(r.regions(128, 64), Regions::Whole);
        assert_eq!(r.regions(128, 64), Regions::Whole);
        r.on_sent(&f);
        let f = refine_frame(&[(0, 0, 2, 2)], &[]);
        r.on_captured(&f);
        assert_eq!(r.regions(128, 64), Regions::Tiles(vec![tile(0, 0, 1)]));
    }

    #[test]
    fn refine_sent_tiles_become_pending() {
        let mut r = RefineState::default();
        let f = refine_frame(&[], &[]);
        r.on_captured(&f);
        r.on_sent(&f);
        let pending = r.pending.as_ref().unwrap();
        assert_eq!(pending.tiles().count(), 8);
        assert!(r.due.is_some() && r.last_frame.is_some());
        // Dirty tiles of a frame not sent yet are not pending.
        r.pending.as_mut().unwrap().clear();
        let f = refine_frame(&[(40, 40, 2, 2)], &[]);
        r.on_captured(&f);
        assert!(r.pending.as_ref().unwrap().is_empty());
        r.on_sent(&f);
        assert!(r.pending.as_ref().unwrap().is_set(1, 1));
        assert_eq!(r.pending.as_ref().unwrap().tiles().count(), 1);
        assert_eq!(r.regions(128, 64), Regions::Nothing);
    }

    #[test]
    fn refine_chunk_sends_text_and_skips_pictures() {
        let f = refine_frame(&[], &[(1, 0)]);
        let FrameData::Raw(data) = &f.data else {
            unreachable!()
        };
        let mut pending = TileGrid::new(128, 64);
        pending.mark_all();
        let chunk = take_refine_chunk(&mut pending, data, f.stride);
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
            let chunk = take_refine_chunk(&mut pending, &data, (w * 4) as usize);
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
    fn layout_larger_than_the_display_is_the_display_size() {
        assert_eq!(clamp_size(1300, 760, (1280, 720)), (1280, 720));
        assert_eq!(clamp_size(1024, 600, (1280, 720)), (1024, 600));
        assert_eq!(clamp_size(70_000, 600, (1280, 720)), (1280, 600));
    }

    #[test]
    fn keyframe_due_after_frame() {
        // A frame the encoder dropped is not a reference: no keyframe for it ...
        assert!(!keyframe_due(false, false, FrameResult::Dropped));
        // ... but a keyframe that was due (e.g. the dropped frame was the forced
        // keyframe) stays due.
        assert!(keyframe_due(true, false, FrameResult::Dropped));
        // A keyframe asked for (new surface, scene change) and dropped is due.
        assert!(keyframe_due(false, true, FrameResult::Dropped));
        // The encoder moved on without the client: keyframe.
        assert!(keyframe_due(false, false, FrameResult::Lost));
        assert!(keyframe_due(true, false, FrameResult::Lost));
        // The client got the frame.
        assert!(!keyframe_due(true, true, FrameResult::Sent));
        assert!(!keyframe_due(false, false, FrameResult::Sent));
    }

    fn run(level: &mut BitrateLevel, t0: Instant, ms: u64, delay: Option<f32>) -> f64 {
        let d = delay.map(|d| (d, MIN_UP_SAMPLES));
        level.update(d, d, t0 + std::time::Duration::from_millis(ms))
    }

    #[test]
    fn bitrate_level_no_samples_keeps_level() {
        let t0 = Instant::now();
        let mut l = BitrateLevel::default();
        assert_eq!(run(&mut l, t0, 0, None), 1.0);
        assert_eq!(run(&mut l, t0, 10, Some(500.0)), 0.8);
        // Quiet screen: no samples, no change for a while.
        assert_eq!(run(&mut l, t0, 5_000, None), 0.8);
        // Too few samples count as none.
        let few = Some((500.0, 3));
        assert_eq!(
            l.update(few, few, t0 + std::time::Duration::from_secs(9)),
            0.8
        );
    }

    #[test]
    fn bitrate_level_steps_up_after_a_long_quiet() {
        let t0 = Instant::now();
        let mut l = BitrateLevel::default();
        for i in 0..3 {
            run(&mut l, t0, i * 600, Some(500.0));
        }
        assert_eq!(run(&mut l, t0, 1_300, Some(500.0)), 0.3);
        // Quiet after the samples at 1.3 s: one level up for each 10 s.
        assert_eq!(run(&mut l, t0, 2_000, None), 0.3);
        assert_eq!(run(&mut l, t0, 11_200, None), 0.3);
        assert_eq!(run(&mut l, t0, 11_300, None), 0.5);
        assert_eq!(run(&mut l, t0, 21_000, None), 0.5);
        assert_eq!(run(&mut l, t0, 21_300, None), 0.8);
        // Motion again with a full link: down at once (probe failed).
        assert_eq!(run(&mut l, t0, 21_400, Some(500.0)), 0.5);
    }

    #[test]
    fn bitrate_level_counts_quiet_time_without_calls() {
        // On a quiet screen no frame is sent and nothing calls `update`.
        let t0 = Instant::now();
        let mut l = BitrateLevel::default();
        for i in 0..3 {
            run(&mut l, t0, i * 600, Some(500.0));
        }
        assert_eq!(run(&mut l, t0, 1_300, Some(500.0)), 0.3);
        assert_eq!(run(&mut l, t0, 21_400, None), 0.8);
        assert_eq!(run(&mut l, t0, 22_000, None), 0.8);
        assert_eq!(run(&mut l, t0, 31_300, None), 1.0);
    }

    #[test]
    fn bitrate_level_ignores_small_queues() {
        let t0 = Instant::now();
        let mut l = BitrateLevel::default();
        // Wi-Fi jitter and a client that decodes slowly stay below the limit.
        for i in 0..600u64 {
            let d = [5.0, 45.0, 90.0][(i % 3) as usize];
            assert_eq!(run(&mut l, t0, i * 16, Some(d)), 1.0);
        }
    }

    #[test]
    fn bitrate_level_steps_down_once_per_hold() {
        let t0 = Instant::now();
        let mut l = BitrateLevel::default();
        assert_eq!(run(&mut l, t0, 0, Some(300.0)), 0.8);
        assert_eq!(run(&mut l, t0, 400, Some(300.0)), 0.8);
        assert_eq!(run(&mut l, t0, 500, Some(300.0)), 0.5);
        assert_eq!(run(&mut l, t0, 1_000, Some(300.0)), 0.3);
        assert_eq!(run(&mut l, t0, 1_500, Some(300.0)), 0.15);
        assert_eq!(run(&mut l, t0, 2_000, Some(300.0)), 0.15);
    }

    #[test]
    fn bitrate_level_steps_up_after_dwell() {
        let t0 = Instant::now();
        let mut l = BitrateLevel::default();
        run(&mut l, t0, 0, Some(300.0));
        run(&mut l, t0, 500, Some(300.0));
        assert_eq!(run(&mut l, t0, 1_100, Some(10.0)), 0.5);
        assert_eq!(run(&mut l, t0, 4_000, Some(10.0)), 0.5);
        // One level per dwell.
        assert_eq!(run(&mut l, t0, 4_100, Some(10.0)), 0.8);
        // A sample in the band between the limits restarts the dwell.
        assert_eq!(run(&mut l, t0, 5_000, Some(70.0)), 0.8);
        assert_eq!(run(&mut l, t0, 7_500, Some(10.0)), 0.8);
        assert_eq!(run(&mut l, t0, 10_400, Some(10.0)), 0.8);
        assert_eq!(run(&mut l, t0, 10_500, Some(10.0)), 1.0);
    }

    #[test]
    fn bitrate_level_failed_probe_doubles_dwell() {
        let t0 = Instant::now();
        let mut l = BitrateLevel::default();
        assert_eq!(run(&mut l, t0, 0, Some(300.0)), 0.8);
        assert_eq!(run(&mut l, t0, 100, Some(10.0)), 0.8);
        assert_eq!(run(&mut l, t0, 3_100, Some(10.0)), 1.0);
        // The step up congests the link again: back down, and wait 6 s next time.
        assert_eq!(run(&mut l, t0, 4_000, Some(300.0)), 0.8);
        assert_eq!(run(&mut l, t0, 4_100, Some(10.0)), 0.8);
        assert_eq!(run(&mut l, t0, 9_000, Some(10.0)), 0.8);
        assert_eq!(run(&mut l, t0, 10_100, Some(10.0)), 1.0);
        // 30 s without a step down: the dwell is 3 s again.
        assert_eq!(run(&mut l, t0, 40_000, Some(300.0)), 0.8);
        assert_eq!(run(&mut l, t0, 40_100, Some(10.0)), 0.8);
        assert_eq!(run(&mut l, t0, 43_100, Some(10.0)), 1.0);
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

    /// Evaluate one frame, 16 ms after the previous one (60 fps).
    fn ev(ramp: &mut ProgressiveRamp, dirty: u64, total: u64, base: u32) -> ProgressiveAction {
        let now = match ramp.last_frame {
            Some(t) => t + std::time::Duration::from_millis(16),
            None => Instant::now(),
        };
        ramp.evaluate(dirty, total, base, now)
    }

    #[test]
    fn test_progressive_no_trigger_below_threshold() {
        let mut ramp = ProgressiveRamp::new();
        // 20% dirty — below 30% threshold
        let action = ev(&mut ramp, 200, 1000, 8_000_000);
        assert_eq!(action, ProgressiveAction::None);
        assert!(!ramp.is_ramping());
    }

    #[test]
    fn test_progressive_triggers_on_scene_change() {
        let mut ramp = ProgressiveRamp::new();
        // 50% dirty — above threshold
        let action = ev(&mut ramp, 500, 1000, 8_000_000);
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
        let action = ev(&mut ramp, 1000, 1000, base);
        assert!(matches!(action, ProgressiveAction::SceneChange { .. }));

        // Ramp frames 1..4 should produce increasing bitrates
        let mut prev_bitrate = 0u32;
        for i in 0..4 {
            let action = ev(&mut ramp, 0, 1000, base);
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
        let action = ev(&mut ramp, 0, 1000, base);
        assert_eq!(action, ProgressiveAction::RampComplete { bitrate: base });
        assert!(!ramp.is_ramping());
    }

    #[test]
    fn test_progressive_no_retrigger_during_ramp() {
        let mut ramp = ProgressiveRamp::new();
        // Trigger
        ev(&mut ramp, 1000, 1000, 8_000_000);
        // Another scene change during ramp should NOT re-trigger
        let action = ev(&mut ramp, 1000, 1000, 8_000_000);
        assert!(matches!(action, ProgressiveAction::Ramp { .. }));
    }

    #[test]
    fn test_progressive_retrigger_after_ramp_complete() {
        let mut ramp = ProgressiveRamp::new();
        let base = 8_000_000;
        // Complete a full ramp
        ev(&mut ramp, 1000, 1000, base);
        for _ in 0..5 {
            ev(&mut ramp, 0, 1000, base);
        }
        assert!(!ramp.is_ramping());

        // Should be able to trigger again
        let action = ev(&mut ramp, 1000, 1000, base);
        assert!(matches!(action, ProgressiveAction::SceneChange { .. }));
    }

    #[test]
    fn test_progressive_sustained_motion_is_one_scene_change() {
        let mut ramp = ProgressiveRamp::new();
        let base = 8_000_000;
        let scene_changes = (0..120)
            .filter(|_| {
                matches!(
                    ev(&mut ramp, 1000, 1000, base),
                    ProgressiveAction::SceneChange { .. }
                )
            })
            .count();
        assert_eq!(scene_changes, 1);
        assert!(!ramp.is_ramping());
    }

    #[test]
    fn test_progressive_rearm_after_quiet_gap() {
        let mut ramp = ProgressiveRamp::new();
        let base = 8_000_000;
        for _ in 0..10 {
            ev(&mut ramp, 1000, 1000, base);
        }
        // No frame for a while (static screen), then a large change.
        let later = ramp.last_frame.unwrap() + PROGRESSIVE_REARM_GAP;
        let action = ramp.evaluate(1000, 1000, base, later);
        assert!(matches!(action, ProgressiveAction::SceneChange { .. }));
    }

    #[test]
    fn test_progressive_minimum_bitrate_floor() {
        let mut ramp = ProgressiveRamp::new();
        // Very low base bitrate: 1 Mbps * 0.25 = 250 Kbps, should clamp to 500 Kbps
        let action = ev(&mut ramp, 1000, 1000, 1_000_000);
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
        let action = ev(&mut ramp, 0, 0, 8_000_000);
        assert_eq!(action, ProgressiveAction::None);
    }

    #[test]
    fn test_progressive_exact_threshold() {
        let mut ramp = ProgressiveRamp::new();
        // Exactly 30% should trigger
        let action = ev(&mut ramp, 300, 1000, 8_000_000);
        assert!(matches!(action, ProgressiveAction::SceneChange { .. }));
    }

    #[test]
    fn test_progressive_just_below_threshold() {
        let mut ramp = ProgressiveRamp::new();
        // 29.9% should not trigger
        let action = ev(&mut ramp, 299, 1000, 8_000_000);
        assert_eq!(action, ProgressiveAction::None);
    }
}
