use anyhow::Result;
use bytes::Bytes;
use ironrdp_server::{
    BitmapUpdate, DesktopSize, DisplayUpdate, GfxDirtyH264Update, GfxFrameUpdate,
    H264Rect, PixelFormat as RdpPixelFormat, PointerPositionAttribute, RdpServerDisplay,
    RdpServerDisplayUpdates, gfx::GfxState,
};
use macrdp_audio::SharedAudioTx;
use macrdp_capture::{CaptureConfig, CapturePixelFormat, CapturedFrame, CgFallbackCapturer, FrameData, ScreenCapturer, get_cursor_position};
use macrdp_encode::{self, Quality, VideoEncoder, encode_rect_h264};
use std::num::{NonZeroU16, NonZeroUsize};
use std::sync::{Arc, Mutex};

use crate::handler::MouseCoordMapper;
use crate::perf_stats::SharedPerfStats;

/// Maximum tile size for bitmap updates
const TILE_SIZE: u16 = 64;

/// Maximum total dirty area (in pixels) for the uncompressed GFX path.
/// Below this threshold, raw pixels are sent instead of H.264 encoding.
/// 262144 pixels ≈ 512×512 rect ≈ 1 MB of raw BGRA data.
/// Covers most UI interactions (menus, buttons, tooltips, text cursor blinks).
const UNCOMPRESSED_MAX_PIXELS: u32 = 262144;

/// Max dirty rects to encode individually before falling back to full-frame H.264.
const DIRTY_RECT_MAX_COUNT: usize = 16;

/// Dirty area below this fraction of total pixels = "low activity" (0.5%).
const LOW_ACTIVITY_FRACTION: f64 = 0.005;
/// After this many consecutive low-activity frames, start reducing FPS.
const LOW_ACTIVITY_RAMP_FRAMES: u32 = 60;
/// Minimum FPS during sustained low activity.
const LOW_ACTIVITY_MIN_FPS: u32 = 5;

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
    fn evaluate(&mut self, dirty_pixels: u64, total_pixels: u64, base_bitrate: u32) -> ProgressiveAction {
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
                return ProgressiveAction::RampComplete { bitrate: base_bitrate };
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

/// Convert a captured frame into tiled BitmapUpdate chunks
pub fn frame_to_bitmap_updates(frame: &CapturedFrame, tile_size: u16) -> Vec<BitmapUpdate> {
    let bgra = match frame.data.as_bgra_bytes() {
        Some(b) => b,
        None => return Vec::new(), // PixelBuffer frames don't go through bitmap path
    };
    let mut updates = Vec::new();
    let bpp: usize = 4;

    let cols = (frame.width as u16 + tile_size - 1) / tile_size;
    let rows = (frame.height as u16 + tile_size - 1) / tile_size;

    for row in 0..rows {
        for col in 0..cols {
            let x = col * tile_size;
            let y = row * tile_size;
            let w = (frame.width as u16 - x).min(tile_size);
            let h = (frame.height as u16 - y).min(tile_size);

            let Some(width) = NonZeroU16::new(w) else { continue };
            let Some(height) = NonZeroU16::new(h) else { continue };

            let mut tile_data = Vec::with_capacity(w as usize * h as usize * bpp);
            for dy in 0..h {
                let src_y = (y + dy) as usize;
                let src_x_start = x as usize * bpp;
                let src_x_end = src_x_start + w as usize * bpp;
                let row_start = src_y * frame.stride;
                if row_start + src_x_end <= bgra.len() {
                    tile_data.extend_from_slice(&bgra[row_start + src_x_start..row_start + src_x_end]);
                }
            }

            let stride = w as usize * bpp;
            let Some(stride) = NonZeroUsize::new(stride) else { continue };

            updates.push(BitmapUpdate {
                x,
                y,
                width,
                height,
                format: RdpPixelFormat::BgrA32,
                data: Bytes::from(tile_data),
                stride,
            });
        }
    }

    updates
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
    show_cursor: bool,
    base_bitrate: u32,
    gfx_state: Arc<Mutex<GfxState>>,
    /// Shared audio sender slot — updated per connection by AudioFactory
    shared_audio_tx: Option<SharedAudioTx>,
    /// Shared performance statistics collector (None = disabled)
    perf_stats: Option<SharedPerfStats>,
}

impl MacDisplay {
    pub fn new(
        width: u16, height: u16,
        fixed_resolution: bool,
        frame_rate: u32, quality: Quality,
        encoder_pref: macrdp_encode::EncoderPreference,
        mode_444: bool,
        show_cursor: bool,
        bitrate_override: Option<u32>,
        gfx_state: Arc<Mutex<GfxState>>,
        coord_mapper: MouseCoordMapper,
        shared_audio_tx: Option<SharedAudioTx>,
        perf_stats: Option<SharedPerfStats>,
    ) -> Self {
        let base_bitrate = bitrate_override
            .unwrap_or_else(|| macrdp_encode::screen_bitrate(width as u32, height as u32, frame_rate as f32, quality));
        tracing::info!(base_bitrate_mbps = base_bitrate as f64 / 1_000_000.0, "Base bitrate");
        // Max resolution: use physical display size so clients can resize freely.
        // SCK scales_to_fit handles any resolution.
        let (max_w, max_h) = macrdp_capture::detect_display_scale()
            .map(|scale| (width.max((width as u32 * scale) as u16), height.max((height as u32 * scale) as u16)))
            .unwrap_or((width.max(3840), height.max(2160)));
        Self {
            width, height,
            max_width: max_w, max_height: max_h,
            coord_mapper,
            fixed_resolution,
            frame_rate, quality, encoder_pref, mode_444, show_cursor, base_bitrate, gfx_state,
            shared_audio_tx,
            perf_stats,
        }
    }
}

#[async_trait::async_trait]
impl RdpServerDisplay for MacDisplay {
    async fn size(&mut self) -> DesktopSize {
        DesktopSize { width: self.width, height: self.height }
    }

    fn request_resize(&mut self, width: u16, height: u16) {
        if self.fixed_resolution {
            tracing::debug!("Ignoring resize request — resolution is fixed by config");
            return;
        }
        let w = width.min(self.max_width);
        let h = height.min(self.max_height);
        if w > 0 && h > 0 && (w != self.width || h != self.height) {
            tracing::info!(
                old_w = self.width, old_h = self.height,
                new_w = w, new_h = h,
                "Adopting client-requested resolution"
            );
            self.width = w;
            self.height = h;
            self.base_bitrate = macrdp_encode::screen_bitrate(
                w as u32, h as u32, self.frame_rate as f32, self.quality,
            );
            self.coord_mapper.update_rdp_size(w, h);
        }
    }

    async fn updates(&mut self) -> Result<Box<dyn RdpServerDisplayUpdates>> {
        let capture_config = CaptureConfig {
            width: self.width as u32,
            height: self.height as u32,
            frame_rate: self.frame_rate,
            pixel_format: if self.encoder_pref == macrdp_encode::EncoderPreference::Hardware && !self.mode_444 {
                CapturePixelFormat::Nv12
            } else {
                CapturePixelFormat::Bgra
            },
            show_cursor: self.show_cursor,
        };
        // Read the current audio sender from the shared slot (set by AudioFactory per connection)
        let audio_tx = self.shared_audio_tx.as_ref()
            .and_then(|shared| shared.lock().unwrap().clone());
        let capturer = ScreenCapturer::new(capture_config.clone(), audio_tx).await?;

        // Create H.264 encoder with configured quality and encoder preference
        let encoder = macrdp_encode::create_encoder(
            self.width as u32,
            self.height as u32,
            self.frame_rate as f32,
            self.quality,
            self.encoder_pref,
            self.mode_444,
            self.base_bitrate,
        ).ok();

        if encoder.is_some() {
            tracing::info!("H.264 encoder available — will use GFX path when client supports it");
        }

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
            encoder,
            gfx_state: Arc::clone(&self.gfx_state),
            base_bitrate: self.base_bitrate,
            mode_444: self.mode_444,
            display_frame_count: 0,
            skip_next_frame: false,
            overload_count: 0,
            perf_stats: self.perf_stats.clone(),
            low_activity_frames: 0,
            current_fps: self.frame_rate,
            cursor_interval,
            last_cursor_pos: (0, 0),
            cursor_initialized: false,
            last_applied_bitrate: self.base_bitrate,
            progressive: ProgressiveRamp::new(),
        }))
    }
}

struct MacDisplayUpdates {
    capturer: ScreenCapturer,
    capture_config: CaptureConfig,
    encoder: Option<Box<dyn VideoEncoder>>,
    gfx_state: Arc<Mutex<GfxState>>,
    base_bitrate: u32,
    mode_444: bool,
    display_frame_count: u64,
    skip_next_frame: bool,
    /// Counter for rate-limiting encode overload warnings
    overload_count: u64,
    /// Shared performance statistics collector (None = disabled)
    perf_stats: Option<SharedPerfStats>,
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
    /// Last bitrate applied to the encoder (to avoid redundant set_bitrate calls)
    last_applied_bitrate: u32,
    progressive: ProgressiveRamp,
}

#[async_trait::async_trait]
impl RdpServerDisplayUpdates for MacDisplayUpdates {
    async fn next_update(&mut self) -> Result<Option<DisplayUpdate>> {
        // Send initial DefaultPointer when cursor channel is active
        if self.cursor_interval.is_some() && !self.cursor_initialized {
            self.cursor_initialized = true;
            let (x, y) = get_cursor_position();
            self.last_cursor_pos = (x, y);
            tracing::info!("Cursor channel active — sending initial pointer at ({x}, {y})");
            return Ok(Some(DisplayUpdate::DefaultPointer));
        }

        if self.cursor_interval.is_none() {
            let frame = match self.next_frame().await {
                Some(f) => f,
                None => return Ok(None),
            };
            return self.encode_and_send(frame);
        }

        loop {
            // Take interval out to avoid borrowing self through it during select!
            let mut cursor_iv = self.cursor_interval.take().unwrap();
            let result = tokio::select! {
                biased;
                _ = cursor_iv.tick() => {
                    self.cursor_interval = Some(cursor_iv);
                    let (x, y) = get_cursor_position();
                    if (x, y) != self.last_cursor_pos {
                        self.last_cursor_pos = (x, y);
                        return Ok(Some(DisplayUpdate::PointerPosition(
                            PointerPositionAttribute { x, y }
                        )));
                    }
                    continue;
                }
                frame_result = self.next_frame() => {
                    self.cursor_interval = Some(cursor_iv);
                    frame_result
                }
            };
            return match result {
                Some(frame) => self.encode_and_send(frame),
                None => Ok(None),
            };
        }
    }
}

impl MacDisplayUpdates {
    /// Wait for the next video frame, handling SCK recovery and frame draining.
    async fn next_frame(&mut self) -> Option<CapturedFrame> {
        loop {
            let event = match self.capturer.next_frame().await {
                Some(e) => e,
                None => {
                    tracing::warn!("SCStream stopped — switching to CoreGraphics fallback (lock screen?)");
                    let fallback = CgFallbackCapturer::new(&self.capture_config);
                    loop {
                        match ScreenCapturer::new(self.capture_config.clone(), None).await {
                            Ok(new_capturer) => {
                                tracing::info!("SCStream recovered — switching back from CoreGraphics");
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
            match self.capturer.try_next_frame() {
                Some(macrdp_capture::CaptureEvent::Frame(_newer)) => continue,
                Some(macrdp_capture::CaptureEvent::Idle) => return Some(frame),
                None => return Some(frame),
            }
        }
    }

    /// Reduce capture FPS when dirty area is consistently tiny, restore instantly on activity.
    fn adjust_fps_for_activity(&mut self, frame: &CapturedFrame) {
        let total_pixels = frame.width as u64 * frame.height as u64;
        if total_pixels == 0 || frame.dirty_rects.is_empty() {
            return;
        }

        let dirty_pixels: u64 = frame.dirty_rects.iter()
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
        let dirty_pixels: u64 = frame.dirty_rects.iter()
            .map(|r| r.width as u64 * r.height as u64)
            .sum();

        match self.progressive.evaluate(dirty_pixels, total_pixels, self.last_applied_bitrate) {
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

    fn encode_and_send(&mut self, frame: CapturedFrame) -> Result<Option<DisplayUpdate>> {
        // Encode overload protection: skip this frame if previous encode took too long
        if self.skip_next_frame {
            self.skip_next_frame = false;
            return Ok(Some(DisplayUpdate::DefaultPointer));
        }

        // Activity-based FPS reduction
        self.adjust_fps_for_activity(&frame);

        // Check GFX state and AVC444 negotiation
        let (gfx_ready, use_444) = {
            let state = self.gfx_state.lock().unwrap();
            let ready = state.is_ready() && self.encoder.is_some();
            let use_444 = self.mode_444
                && state.avc444_supported
                && state.avc444_enabled;
            (ready, use_444)
        };

        if gfx_ready {
            // Adaptive bitrate: adjust encoder bitrate based on network conditions
            if let Some(encoder) = &mut self.encoder {
                let adaptive = {
                    let state = self.gfx_state.lock().unwrap();
                    state.adaptive_bitrate(self.base_bitrate)
                };
                if adaptive != self.last_applied_bitrate
                    && (adaptive as f64 - self.last_applied_bitrate as f64).abs()
                        > self.base_bitrate as f64 * 0.1
                {
                    tracing::info!(
                        from_mbps = self.last_applied_bitrate as f64 / 1_000_000.0,
                        to_mbps = adaptive as f64 / 1_000_000.0,
                        "Adaptive bitrate adjustment"
                    );
                    encoder.set_bitrate(adaptive);
                    self.last_applied_bitrate = adaptive;
                }
            }

            // Progressive rendering: detect scene change and apply quality ramp
            self.apply_progressive_bitrate(&frame);

            // GFX dirty-rect path — for small dirty regions, encode each rect
            // independently with OpenH264 instead of full-frame H.264
            if !frame.dirty_rects.is_empty() {
                let total_area: u32 = frame.dirty_rects.iter()
                    .map(|r| r.width * r.height)
                    .sum();

                if total_area > 0 && total_area <= UNCOMPRESSED_MAX_PIXELS
                    && frame.dirty_rects.len() <= DIRTY_RECT_MAX_COUNT
                {
                    if let Some(bgra) = frame.data.as_bgra_bytes() {
                        let t0 = std::time::Instant::now();
                        let mut h264_rects: Vec<H264Rect> = Vec::new();
                        let mut encode_failed = false;

                        for r in frame.dirty_rects.iter().filter(|r| r.width > 0 && r.height > 0) {
                            let x = r.x.min(frame.width.saturating_sub(1));
                            let y = r.y.min(frame.height.saturating_sub(1));
                            let w = r.width.min(frame.width - x);
                            let h = r.height.min(frame.height - y);
                            if w == 0 || h == 0 { continue; }

                            match encode_rect_h264(bgra, frame.stride, x, y, w, h) {
                                Ok(encoded) => {
                                    h264_rects.push(H264Rect {
                                        x: x as u16,
                                        y: y as u16,
                                        width: w as u16,
                                        height: h as u16,
                                        enc_width: encoded.width as u16,
                                        enc_height: encoded.height as u16,
                                        h264_data: encoded.data,
                                    });
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        rect_x = r.x, rect_y = r.y,
                                        rect_w = w, rect_h = h,
                                        "Dirty-rect H.264 encode failed: {e:#}, falling back to full frame"
                                    );
                                    encode_failed = true;
                                    break;
                                }
                            }
                        }

                        if !encode_failed && !h264_rects.is_empty() {
                            self.display_frame_count += 1;
                            let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
                            let total_h264: usize = h264_rects.iter().map(|r| r.h264_data.len()).sum();
                            {
                                let mut st = self.gfx_state.lock().unwrap();
                                st.last_encode_ms = encode_ms;
                                st.last_frame_bytes = total_h264 as u32;
                            }
                            tracing::debug!(
                                display_frame = self.display_frame_count,
                                rects = h264_rects.len(),
                                total_pixels = total_area,
                                h264_bytes = total_h264,
                                encode_ms = format!("{:.1}", encode_ms),
                                "Display: sending GFX dirty-rect H.264"
                            );
                            return Ok(Some(DisplayUpdate::GfxDirtyH264(GfxDirtyH264Update {
                                rects: h264_rects,
                                width: frame.width as u16,
                                height: frame.height as u16,
                            })));
                        }
                    }
                    // PixelBuffer frames or encode failure — fall through to full-frame H.264
                }
            }

            // GFX H.264 path — always send at capture rate, never block on acks
            if let Some(encoder) = &mut self.encoder {
                self.display_frame_count += 1;
                let t0 = std::time::Instant::now();

                // Route based on frame data type
                match &frame.data {
                    FrameData::PixelBuffer(buf) => {
                        // Zero-copy VT path — encode CVPixelBuffer directly
                        match encoder.encode_pixel_buffer(buf.as_ptr(), false) {
                            Ok(encoded) if !encoded.data.is_empty() => {
                                let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
                                let is_keyframe = encoded.is_keyframe;
                                tracing::debug!(
                                    display_frame = self.display_frame_count,
                                    h264_bytes = encoded.data.len(),
                                    is_keyframe,
                                    encode_ms = format!("{:.1}", encode_ms),
                                    "Display: sending zero-copy GFX frame"
                                );
                                {
                                    let mut st = self.gfx_state.lock().unwrap();
                                    st.last_encode_ms = encode_ms;
                                    st.last_frame_bytes = encoded.data.len() as u32;
                                }
                                if let Some(ps) = &self.perf_stats {
                                    ps.lock().unwrap().record_frame(encode_ms, encoded.data.len() as u32, is_keyframe);
                                }
                                let frame_interval_ms = 1000.0 / self.capture_config.frame_rate as f64;
                                if encode_ms > frame_interval_ms * 0.95 {
                                    self.skip_next_frame = true;
                                    self.overload_count += 1;
                                    // Rate-limit log: warn once per ~60 overloads, debug otherwise
                                    if self.overload_count % 60 == 1 {
                                        tracing::warn!(
                                            encode_ms = format!("{:.1}", encode_ms),
                                            frame_interval_ms = format!("{:.1}", frame_interval_ms),
                                            overload_total = self.overload_count,
                                            "encode overload — skipping frames"
                                        );
                                    }
                                }
                                return Ok(Some(DisplayUpdate::GfxFrame(GfxFrameUpdate {
                                    h264_data: encoded.data,
                                    width: frame.width as u16,
                                    height: frame.height as u16,
                                    enc_width: encoded.width as u16,
                                    enc_height: encoded.height as u16,
                                    is_keyframe,
                                    h264_aux: None,
                                })));
                            }
                            Ok(_) => {
                                tracing::warn!("Zero-copy encode returned empty data");
                                return Ok(Some(DisplayUpdate::DefaultPointer));
                            }
                            Err(e) => {
                                tracing::warn!("Zero-copy encode failed: {e}, falling back to DefaultPointer");
                                return Ok(Some(DisplayUpdate::DefaultPointer));
                            }
                        }
                    }
                    FrameData::Raw(_) => {
                        // Existing BGRA encode path — continues below
                    }
                }
                let bgra = frame.data.as_bgra_bytes().unwrap();

                // AVC444 dual-stream path
                if use_444 && encoder.supports_444() {
                    match encoder.encode_bgra_444(bgra, frame.width, frame.height, frame.stride) {
                        Ok(encoded) if !encoded.main_view.data.is_empty() => {
                            let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
                            let total_bytes = encoded.main_view.data.len() + encoded.aux_view.data.len();
                            let is_keyframe = encoded.main_view.is_keyframe;
                            tracing::debug!(
                                display_frame = self.display_frame_count,
                                main_bytes = encoded.main_view.data.len(),
                                aux_bytes = encoded.aux_view.data.len(),
                                is_keyframe,
                                encode_ms = format!("{:.1}", encode_ms),
                                "Display: sending AVC444 GFX frame"
                            );
                            {
                                let mut st = self.gfx_state.lock().unwrap();
                                st.last_encode_ms = encode_ms;
                                st.last_frame_bytes = total_bytes as u32;
                            }
                            if let Some(ps) = &self.perf_stats {
                                ps.lock().unwrap().record_frame(encode_ms, total_bytes as u32, is_keyframe);
                            }
                            let frame_interval_ms = 1000.0 / self.capture_config.frame_rate as f64;
                            if encode_ms > frame_interval_ms * 0.95 {
                                self.skip_next_frame = true;
                                self.overload_count += 1;
                                if self.overload_count % 60 == 1 {
                                    tracing::warn!(
                                        encode_ms = format!("{:.1}", encode_ms),
                                        frame_interval_ms = format!("{:.1}", frame_interval_ms),
                                        overload_total = self.overload_count,
                                        "encode overload — skipping frames"
                                    );
                                }
                            }
                            return Ok(Some(DisplayUpdate::GfxFrame(GfxFrameUpdate {
                                h264_data: encoded.main_view.data,
                                width: frame.width as u16,
                                height: frame.height as u16,
                                enc_width: encoded.main_view.width as u16,
                                enc_height: encoded.main_view.height as u16,
                                is_keyframe,
                                h264_aux: Some(encoded.aux_view.data),
                            })));
                        }
                        Ok(_) => {
                            tracing::warn!(
                                display_frame = self.display_frame_count,
                                "AVC444 encode returned EMPTY data — frame dropped!"
                            );
                            return Ok(Some(DisplayUpdate::DefaultPointer));
                        }
                        Err(e) => {
                            tracing::warn!(display_frame = self.display_frame_count, "AVC444 encode failed: {e}, falling back to AVC420");
                            // Fall through to AVC420 path below
                        }
                    }
                }

                // AVC420 path (default or fallback from AVC444 failure)
                match encoder.encode_bgra(bgra, frame.width, frame.height, frame.stride) {
                    Ok(encoded) if !encoded.data.is_empty() => {
                        let encode_ms = t0.elapsed().as_secs_f64() * 1000.0;
                        let is_keyframe = encoded.is_keyframe;
                        tracing::debug!(
                            display_frame = self.display_frame_count,
                            h264_bytes = encoded.data.len(),
                            is_keyframe,
                            encode_ms = format!("{:.1}", encode_ms),
                            "Display: sending GFX frame"
                        );
                        {
                            let mut st = self.gfx_state.lock().unwrap();
                            st.last_encode_ms = encode_ms;
                            st.last_frame_bytes = encoded.data.len() as u32;
                        }
                        if let Some(ps) = &self.perf_stats {
                            ps.lock().unwrap().record_frame(encode_ms, encoded.data.len() as u32, is_keyframe);
                        }
                        let frame_interval_ms = 1000.0 / self.capture_config.frame_rate as f64;
                        if encode_ms > frame_interval_ms * 0.95 {
                            self.skip_next_frame = true;
                            self.overload_count += 1;
                            if self.overload_count % 60 == 1 {
                                tracing::warn!(
                                    encode_ms = format!("{:.1}", encode_ms),
                                    frame_interval_ms = format!("{:.1}", frame_interval_ms),
                                    overload_total = self.overload_count,
                                    "encode overload — skipping frames"
                                );
                            }
                        }
                        return Ok(Some(DisplayUpdate::GfxFrame(GfxFrameUpdate {
                            h264_data: encoded.data,
                            width: frame.width as u16,
                            height: frame.height as u16,
                            enc_width: encoded.width as u16,
                            enc_height: encoded.height as u16,
                            is_keyframe,
                            h264_aux: None,
                        })));
                    }
                    Ok(_) => {
                        tracing::warn!(
                            display_frame = self.display_frame_count,
                            "H.264 encode returned EMPTY data — frame dropped!"
                        );
                        return Ok(Some(DisplayUpdate::DefaultPointer));
                    }
                    Err(e) => {
                        tracing::warn!(display_frame = self.display_frame_count, "H.264 encode failed: {e}");
                    }
                }
            }
        } else if self.encoder.is_some() {
            // H.264 encoder exists — never send bitmaps, wait for GFX to become ready.
            // Mixing bitmap and GFX causes 0xd06 DECOMPRESSION_FAILED on reconnect.
            return Ok(Some(DisplayUpdate::DefaultPointer));
        }

        // Bitmap path (only when GFX is not available at all)
        // Requires BGRA raw bytes — PixelBuffer frames should not reach here
        let bgra_bitmap = match &frame.data {
            FrameData::Raw(bytes) => bytes,
            FrameData::PixelBuffer(_) => {
                tracing::warn!("PixelBuffer frame in bitmap path — should not happen");
                return Ok(Some(DisplayUpdate::DefaultPointer));
            }
        };

        if !frame.dirty_rects.is_empty() {
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
                let Some(width) = NonZeroU16::new(w as u16) else { return Ok(None) };
                let Some(height) = NonZeroU16::new(h as u16) else { return Ok(None) };

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

                let Some(stride) = NonZeroUsize::new(dirty_stride) else { return Ok(None) };

                let update = BitmapUpdate {
                    x: min_x as u16,
                    y: min_y as u16,
                    width,
                    height,
                    format: RdpPixelFormat::BgrA32,
                    data: Bytes::from(dirty_data),
                    stride,
                };

                return Ok(Some(DisplayUpdate::Bitmap(update)));
            }
        }

        // No dirty rects available — send full frame (first frame or fallback)
        let Some(width) = NonZeroU16::new(frame.width as u16) else { return Ok(None) };
        let Some(height) = NonZeroU16::new(frame.height as u16) else { return Ok(None) };
        let Some(stride) = NonZeroUsize::new(frame.stride) else { return Ok(None) };

        let update = BitmapUpdate {
            x: 0,
            y: 0,
            width,
            height,
            format: RdpPixelFormat::BgrA32,
            data: bgra_bitmap.clone(),
            stride,
        };

        Ok(Some(DisplayUpdate::Bitmap(update)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_frame_to_bitmap_updates() {
        let frame = CapturedFrame {
            width: 100,
            height: 50,
            data: FrameData::Raw(Bytes::from(vec![0u8; 100 * 50 * 4])),
            stride: 400,
            timestamp_us: 0,
            dirty_rects: vec![],
        };

        let updates = frame_to_bitmap_updates(&frame, 64);
        assert_eq!(updates.len(), 2);
        assert_eq!(updates[0].x, 0);
        assert_eq!(updates[0].width.get(), 64);
        assert_eq!(updates[1].x, 64);
        assert_eq!(updates[1].width.get(), 36);
    }

    #[test]
    fn test_frame_to_bitmap_updates_exact_tile() {
        let frame = CapturedFrame {
            width: 128,
            height: 64,
            data: FrameData::Raw(Bytes::from(vec![0u8; 128 * 64 * 4])),
            stride: 512,
            timestamp_us: 0,
            dirty_rects: vec![],
        };

        let updates = frame_to_bitmap_updates(&frame, 64);
        assert_eq!(updates.len(), 2);
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
                    assert!(bitrate > prev_bitrate, "frame {i}: {bitrate} should be > {prev_bitrate}");
                    assert!(bitrate < base, "frame {i}: {bitrate} should be < base {base}");
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
