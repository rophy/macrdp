use bytes::Bytes;
use ironrdp_displaycontrol::pdu::DisplayControlMonitorLayout;
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
use macrdp_encode::{self, Quality};
use std::num::{NonZeroU16, NonZeroUsize};
use std::sync::{Arc, Mutex};

use crate::handler::MouseCoordMapper;
use crate::perf_stats::SharedPerfStats;

/// Dirty area below this fraction of total pixels = "low activity" (0.5%).
const LOW_ACTIVITY_FRACTION: f64 = 0.005;
/// After this many consecutive low-activity frames, start reducing FPS.
const LOW_ACTIVITY_RAMP_FRAMES: u32 = 60;
/// Minimum FPS during sustained low activity.
const LOW_ACTIVITY_MIN_FPS: u32 = 5;

/// Progressive rendering: on scene changes (large dirty area), temporarily reduce
/// encoder bitrate for fast initial delivery, then ramp quality back over several frames.
#[allow(dead_code)] // reason: video is bitmap only for now; H.264 on EGFX uses this again
#[derive(Debug)]
struct ProgressiveRamp {
    trigger_fraction: f64,
    ramp_frames: u32,
    initial_quality: f32,
    remaining: u32,
}

#[allow(dead_code)] // reason: video is bitmap only for now; H.264 on EGFX uses this again
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

#[allow(dead_code)] // reason: video is bitmap only for now; H.264 on EGFX uses this again
#[derive(Debug, PartialEq)]
enum ProgressiveAction {
    None,
    SceneChange { bitrate: u32, dirty_pct: f64 },
    Ramp { bitrate: u32 },
    RampComplete { bitrate: u32 },
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
    /// H.264 settings: unused while video is bitmap only; H.264 on EGFX uses them again.
    #[allow(dead_code)]
    encoder_pref: macrdp_encode::EncoderPreference,
    /// Whether AVC444 mode is requested by config
    #[allow(dead_code)]
    mode_444: bool,
    show_cursor: bool,
    #[allow(dead_code)]
    base_bitrate: u32,
    /// New desktop size from a Display Control layout, sent as a Resize update
    pending_resize: Arc<Mutex<Option<(u16, u16)>>>,
    /// Shared audio sender slot — updated per connection by AudioFactory
    shared_audio_tx: Option<SharedAudioTx>,
    /// Shared performance statistics collector (None = disabled)
    #[allow(dead_code)]
    perf_stats: Option<SharedPerfStats>,
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
            pending_resize: Arc::new(Mutex::new(None)),
            shared_audio_tx,
            perf_stats,
        }
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
        *self.pending_resize.lock().unwrap() = Some((self.width, self.height));
    }

    async fn updates(&mut self) -> ServerResult<Box<dyn RdpServerDisplayUpdates>> {
        let capture_config = CaptureConfig {
            width: self.width as u32,
            height: self.height as u32,
            frame_rate: self.frame_rate,
            // Bitmap updates need BGRA pixels.
            pixel_format: CapturePixelFormat::Bgra,
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
            pending_resize: Arc::clone(&self.pending_resize),
            low_activity_frames: 0,
            current_fps: self.frame_rate,
            cursor_interval,
            last_cursor_pos: (0, 0),
            cursor_initialized: false,
        }))
    }
}

struct MacDisplayUpdates {
    capturer: ScreenCapturer,
    capture_config: CaptureConfig,
    /// New desktop size from a Display Control layout, sent as a Resize update
    pending_resize: Arc<Mutex<Option<(u16, u16)>>>,
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
}

#[async_trait::async_trait]
impl RdpServerDisplayUpdates for MacDisplayUpdates {
    async fn next_update(&mut self) -> ServerResult<Option<DisplayUpdate>> {
        if let Some((w, h)) = self.pending_resize.lock().unwrap().take() {
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

    fn encode_and_send(&mut self, frame: CapturedFrame) -> ServerResult<Option<DisplayUpdate>> {
        // Activity-based FPS reduction
        self.adjust_fps_for_activity(&frame);

        // Bitmap path: requires BGRA raw bytes — PixelBuffer frames should not reach here
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
                let Some(width) = NonZeroU16::new(w as u16) else {
                    return Ok(None);
                };
                let Some(height) = NonZeroU16::new(h as u16) else {
                    return Ok(None);
                };

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

                let Some(stride) = NonZeroUsize::new(dirty_stride) else {
                    return Ok(None);
                };

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
        let Some(width) = NonZeroU16::new(frame.width as u16) else {
            return Ok(None);
        };
        let Some(height) = NonZeroU16::new(frame.height as u16) else {
            return Ok(None);
        };
        let Some(stride) = NonZeroUsize::new(frame.stride) else {
            return Ok(None);
        };

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
