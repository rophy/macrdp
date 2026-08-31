//! macOS screen capture via ScreenCaptureKit

use std::ffi::c_void;
use std::sync::Arc;

use anyhow::{Context, Result};
use bytes::Bytes;
use core_graphics::access::ScreenCaptureAccess;
use screencapturekit::cv::{CVPixelBuffer, CVPixelBufferLockFlags};
use screencapturekit::prelude::*;
use screencapturekit::stream::configuration::{AudioChannelCount, AudioSampleRate};
use tokio::sync::mpsc;

/// Check if Screen Recording permission is granted (no prompt)
pub fn check_screen_recording_permission() -> bool {
    ScreenCaptureAccess::default().preflight()
}

/// Request Screen Recording permission (triggers system dialog if not granted)
/// Returns true if already granted. Note: even after granting, the app
/// may need to be restarted for the permission to take effect.
pub fn request_screen_recording_permission() -> bool {
    ScreenCaptureAccess::default().request()
}

/// Open System Settings to Privacy & Security page
pub fn open_screen_recording_settings() {
    let _ = std::process::Command::new("open")
        .arg("x-apple.systempreferences:com.apple.preference.security?Privacy_ScreenCapture")
        .spawn();
}

/// A rectangle region
#[derive(Clone, Debug)]
pub struct Rect {
    pub x: u32,
    pub y: u32,
    pub width: u32,
    pub height: u32,
}

/// Pixel format for screen capture output
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum CapturePixelFormat {
    /// BGRA 32-bit (default, needed for OpenH264 and bitmap fallback path)
    Bgra,
    /// NV12 (420f full-range) — zero-copy to VideoToolbox, no color conversion needed
    Nv12,
}

/// Frame pixel data — either raw BGRA bytes or a zero-copy CVPixelBuffer reference
#[derive(Debug)]
pub enum FrameData {
    /// BGRA raw bytes copied from CVPixelBuffer (existing behavior)
    Raw(Bytes),
    /// IOSurface-backed CVPixelBuffer — zero copy, passed directly to VideoToolbox
    PixelBuffer(SafePixelBuffer),
}

impl FrameData {
    /// Get raw BGRA bytes if this is a Raw frame. Returns None for PixelBuffer frames.
    pub fn as_bgra_bytes(&self) -> Option<&[u8]> {
        match self {
            FrameData::Raw(bytes) => Some(bytes),
            FrameData::PixelBuffer(_) => None,
        }
    }
}

/// Events from the screen capture pipeline.
#[derive(Debug)]
pub enum CaptureEvent {
    /// A complete frame with pixel data ready for encoding.
    Frame(CapturedFrame),
    /// Desktop is unchanged — no new pixel data.
    Idle,
}

/// A captured screen frame
#[derive(Debug)]
pub struct CapturedFrame {
    pub width: u32,
    pub height: u32,
    pub data: FrameData,
    /// Bytes per row (valid for FrameData::Raw only)
    pub stride: usize,
    pub timestamp_us: u64,
    /// Regions that changed since the last frame.
    /// Empty means info unavailable — treat as full frame change.
    pub dirty_rects: Vec<Rect>,
}

/// Raw audio frame from ScreenCaptureKit audio callback.
/// Data is interleaved Float32 PCM (L0,R0, L1,R1, ...).
#[derive(Clone)]
pub struct AudioFrame {
    pub data: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
    pub num_samples: usize,
    pub timestamp_ms: u64,
}

/// Configuration for screen capture
#[derive(Clone)]
pub struct CaptureConfig {
    pub width: u32,
    pub height: u32,
    pub frame_rate: u32,
    pub pixel_format: CapturePixelFormat,
    pub show_cursor: bool,
}

/// Screen capturer using ScreenCaptureKit
pub struct ScreenCapturer {
    stream: SCStream,
    frame_rx: mpsc::Receiver<CaptureEvent>,
}

struct VideoOutputHandler {
    frame_tx: mpsc::Sender<CaptureEvent>,
    pixel_format: CapturePixelFormat,
}

struct AudioOutputHandler {
    audio_tx: Arc<mpsc::Sender<AudioFrame>>,
}

impl SCStreamOutputTrait for VideoOutputHandler {
    fn did_output_sample_buffer(&self, sample: CMSampleBuffer, of_type: SCStreamOutputType) {
        if of_type != SCStreamOutputType::Screen {
            return;
        }

        let event = match self.pixel_format {
            CapturePixelFormat::Nv12 => extract_frame_nv12(&sample),
            CapturePixelFormat::Bgra => extract_frame(&sample),
        };
        let Some(event) = event else { return };

        // Non-blocking send — drop frame if channel is full
        let _ = self.frame_tx.try_send(event);
    }
}

impl SCStreamOutputTrait for AudioOutputHandler {
    fn did_output_sample_buffer(&self, sample: CMSampleBuffer, of_type: SCStreamOutputType) {
        if of_type != SCStreamOutputType::Audio {
            return;
        }

        let Some(audio_buffer_list) = sample.audio_buffer_list() else { return };

        let timestamp = sample.presentation_timestamp();
        let timestamp_ms = if timestamp.timescale > 0 {
            ((timestamp.value as u128 * 1000) / timestamp.timescale as u128) as u64
        } else {
            0
        };

        // Get sample rate and channel count from format description
        let (sample_rate, channels) = if let Some(fmt) = sample.format_description() {
            let sr = fmt.audio_sample_rate().unwrap_or(48000.0) as u32;
            let ch = fmt.audio_channel_count().unwrap_or(2) as u16;
            (sr, ch)
        } else {
            (48000u32, 2u16)
        };

        let num_buffers = audio_buffer_list.num_buffers();

        // SCK delivers stereo as a single interleaved buffer (num_buffers == 1, number_channels == 2)
        // or as separate per-channel buffers (num_buffers == 2, number_channels == 1 each).
        let interleaved_data = if num_buffers <= 1 {
            if let Some(buf) = audio_buffer_list.get(0) {
                let raw = buf.data();
                let samples: &[f32] = unsafe {
                    std::slice::from_raw_parts(raw.as_ptr() as *const f32, raw.len() / 4)
                };
                samples.to_vec()
            } else {
                return;
            }
        } else {
            // Non-interleaved: one buffer per channel — interleave manually
            let channel_bufs: Vec<&[f32]> = (0..num_buffers)
                .filter_map(|i| audio_buffer_list.get(i))
                .map(|buf| {
                    let raw = buf.data();
                    unsafe {
                        std::slice::from_raw_parts(raw.as_ptr() as *const f32, raw.len() / 4)
                    }
                })
                .collect();

            if channel_bufs.is_empty() {
                return;
            }
            let n = channel_bufs[0].len();
            let mut out = Vec::with_capacity(n * channel_bufs.len());
            for i in 0..n {
                for ch in &channel_bufs {
                    out.push(ch[i]);
                }
            }
            out
        };

        let num_samples = if channels > 0 {
            interleaved_data.len() / channels as usize
        } else {
            interleaved_data.len()
        };

        let frame = AudioFrame {
            data: interleaved_data,
            sample_rate,
            channels,
            num_samples,
            timestamp_ms,
        };
        let _ = self.audio_tx.try_send(frame);
    }
}

fn extract_frame(sample: &CMSampleBuffer) -> Option<CaptureEvent> {
    use screencapturekit::cm::SCFrameStatus;

    // Idle means desktop is unchanged — propagate as CaptureEvent::Idle.
    // Blank/Suspended/Stopped are non-displayable states — discard silently.
    match sample.frame_status() {
        Some(SCFrameStatus::Idle) => {
            return Some(CaptureEvent::Idle);
        }
        Some(SCFrameStatus::Blank)
        | Some(SCFrameStatus::Suspended)
        | Some(SCFrameStatus::Stopped) => {
            return None;
        }
        _ => {}
    }

    let pixel_buffer: CVPixelBuffer = sample.image_buffer()?;

    let guard = pixel_buffer.lock(CVPixelBufferLockFlags::READ_ONLY).ok()?;

    let width = guard.width() as u32;
    let height = guard.height() as u32;
    let stride = guard.bytes_per_row();
    let pixels = guard.as_slice();

    if width == 0 || height == 0 || pixels.is_empty() {
        return None;
    }

    // Extract dirty rects from the sample buffer
    // Extract dirty rects — screencapturekit's CGRect has x/y/width/height fields
    let dirty_rects = sample
        .dirty_rects()
        .unwrap_or_default()
        .into_iter()
        .filter(|r| r.width > 0.0 && r.height > 0.0)
        .map(|r| Rect {
            x: r.x.max(0.0) as u32,
            y: r.y.max(0.0) as u32,
            width: r.width as u32,
            height: r.height as u32,
        })
        .collect::<Vec<_>>();

    let data = Bytes::copy_from_slice(pixels);

    let t = sample.presentation_timestamp();
    let timestamp_us = if t.timescale > 0 {
        ((t.value as u128 * 1_000_000) / t.timescale as u128) as u64
    } else {
        0
    };

    Some(CaptureEvent::Frame(CapturedFrame {
        width,
        height,
        data: FrameData::Raw(data),
        stride,
        timestamp_us,
        dirty_rects,
    }))
}

/// Extract a frame in NV12 mode — zero-copy CVPixelBuffer wrapped as SafePixelBuffer.
/// The pixel buffer is retained and passed through the channel without locking or copying.
fn extract_frame_nv12(sample: &CMSampleBuffer) -> Option<CaptureEvent> {
    use screencapturekit::cm::SCFrameStatus;

    // Idle means desktop is unchanged — propagate as CaptureEvent::Idle.
    // Blank/Suspended/Stopped are non-displayable states — discard silently.
    match sample.frame_status() {
        Some(SCFrameStatus::Idle) => {
            return Some(CaptureEvent::Idle);
        }
        Some(SCFrameStatus::Blank)
        | Some(SCFrameStatus::Suspended)
        | Some(SCFrameStatus::Stopped) => {
            return None;
        }
        _ => {}
    }

    let pixel_buffer: CVPixelBuffer = sample.image_buffer()?;

    // width()/height() read from the CVPixelBuffer header — no lock required
    let width = pixel_buffer.width() as u32;
    let height = pixel_buffer.height() as u32;

    if width == 0 || height == 0 {
        return None;
    }

    // Extract dirty rects (same logic as the BGRA path)
    let dirty_rects = sample
        .dirty_rects()
        .unwrap_or_default()
        .into_iter()
        .filter(|r| r.width > 0.0 && r.height > 0.0)
        .map(|r| Rect {
            x: r.x.max(0.0) as u32,
            y: r.y.max(0.0) as u32,
            width: r.width as u32,
            height: r.height as u32,
        })
        .collect::<Vec<_>>();

    let t = sample.presentation_timestamp();
    let timestamp_us = if t.timescale > 0 {
        ((t.value as u128 * 1_000_000) / t.timescale as u128) as u64
    } else {
        0
    };

    // Zero-copy: retain the CVPixelBuffer and wrap it as SafePixelBuffer
    let safe_buf = unsafe { SafePixelBuffer::from_raw(pixel_buffer.as_ptr()) };

    Some(CaptureEvent::Frame(CapturedFrame {
        width,
        height,
        data: FrameData::PixelBuffer(safe_buf),
        stride: 0, // Not applicable for NV12 PixelBuffer mode
        timestamp_us,
        dirty_rects,
    }))
}

/// Detect the main display's native scale factor (1 for non-Retina, 2 for Retina).
/// Find the SCK dimensions of the main display (matching CGDisplay::main().id).
/// Falls back to the first SCK display if no match is found.
fn find_main_sc_display_size() -> Result<(u32, u32)> {
    use core_graphics::display::CGDisplay;
    let main_id = CGDisplay::main().id;
    let content = SCShareableContent::get()
        .context("Failed to get shareable content")?;
    let displays = content.displays();
    let display = displays.iter()
        .find(|d| d.display_id() == main_id)
        .or_else(|| displays.first())
        .context("No display found")?;
    Ok((display.width() as u32, display.height() as u32))
}

pub fn detect_display_scale() -> Result<u32> {
    use core_graphics::display::CGDisplay;
    let main = CGDisplay::main();
    let physical_w = main.pixels_wide() as u32;
    let (logical_w, _) = find_main_sc_display_size()?;
    let scale = if logical_w > 0 { physical_w / logical_w } else { 1 };
    Ok(scale.max(1))
}

/// Query the main display's resolution (from ScreenCaptureKit, used for capture sizing)
pub fn detect_display_size() -> Result<(u32, u32)> {
    find_main_sc_display_size()
}

/// Read the current mouse cursor position in CG logical coordinates.
/// Uses CGEventCreate(NULL) which works without Accessibility permission.
pub fn get_cursor_position() -> (u16, u16) {
    extern "C" {
        fn CGEventCreate(source: *const c_void) -> *mut c_void;
        fn CGEventGetLocation(event: *const c_void) -> core_graphics::geometry::CGPoint;
        fn CFRelease(cf: *const c_void);
    }
    unsafe {
        let event = CGEventCreate(std::ptr::null());
        if event.is_null() {
            return (0, 0);
        }
        let point = CGEventGetLocation(event);
        CFRelease(event);
        (point.x.round() as u16, point.y.round() as u16)
    }
}

/// Query the main display's logical bounds from CoreGraphics.
/// This is the coordinate system CGEvent uses, and MUST be used for mouse mapping.
/// May differ from SCDisplay dimensions on non-standard scaling modes.
pub fn detect_cg_display_size() -> Result<(u32, u32)> {
    use core_graphics::display::CGDisplay;
    let main = CGDisplay::main();
    let bounds = main.bounds();
    let w = bounds.size.width as u32;
    let h = bounds.size.height as u32;
    if w == 0 || h == 0 {
        anyhow::bail!("CGDisplay returned zero bounds");
    }
    Ok((w, h))
}

impl ScreenCapturer {
    /// Create a new screen capturer for the main display.
    ///
    /// If `audio_tx` is `Some`, audio capture is enabled and frames are sent
    /// to the provided channel as `AudioFrame` (interleaved Float32 PCM, 48 kHz stereo).
    pub async fn new(
        config: CaptureConfig,
        audio_tx: Option<mpsc::Sender<AudioFrame>>,
    ) -> Result<Self> {
        // SCShareableContent::get() is synchronous, run in blocking task
        let content = tokio::task::spawn_blocking(|| SCShareableContent::get())
            .await?
            .context("Failed to get shareable content (Screen Recording permission needed)")?;

        let main_id = core_graphics::display::CGDisplay::main().id;
        let mut displays = content.displays();
        let main_idx = displays.iter()
            .position(|d| d.display_id() == main_id)
            .unwrap_or(0);
        let display = if displays.is_empty() {
            anyhow::bail!("No display found");
        } else {
            displays.swap_remove(main_idx)
        };

        let actual_width = if config.width == 0 {
            display.width() as u32
        } else {
            config.width
        };
        let actual_height = if config.height == 0 {
            display.height() as u32
        } else {
            config.height
        };

        let filter = SCContentFilter::create()
            .with_display(&display)
            .with_excluding_windows(&[])
            .build();

        let frame_interval = CMTime::new(1, config.frame_rate as i32);

        let captures_audio = audio_tx.is_some();
        let dest_rect = screencapturekit::cg::CGRect::new(
            0.0, 0.0, actual_width as f64, actual_height as f64,
        );
        let stream_config = SCStreamConfiguration::new()
            .with_width(actual_width)
            .with_height(actual_height)
            .with_scales_to_fit(true)
            .with_destination_rect(dest_rect)
            .with_minimum_frame_interval(&frame_interval)
            .with_pixel_format(match config.pixel_format {
                CapturePixelFormat::Nv12 => PixelFormat::YCbCr_420f,
                CapturePixelFormat::Bgra => PixelFormat::BGRA,
            })
            .with_shows_cursor(config.show_cursor)
            .with_captures_audio(captures_audio)
            .with_sample_rate(AudioSampleRate::Rate48000)
            .with_channel_count(AudioChannelCount::Stereo)
            .with_excludes_current_process_audio(true);

        // Channel for capture events: buffer 2 entries to allow for jitter
        let (frame_tx, frame_rx): (mpsc::Sender<CaptureEvent>, mpsc::Receiver<CaptureEvent>) =
            mpsc::channel(2);

        let video_handler = VideoOutputHandler {
            frame_tx,
            pixel_format: config.pixel_format,
        };

        let mut stream = SCStream::new(&filter, &stream_config);
        stream.add_output_handler(video_handler, SCStreamOutputType::Screen);

        // Register audio handler only if a sender was provided
        if let Some(tx) = audio_tx {
            let audio_handler = AudioOutputHandler {
                audio_tx: Arc::new(tx),
            };
            stream.add_output_handler(audio_handler, SCStreamOutputType::Audio);
        }

        stream.start_capture().context("Failed to start capture")?;

        tracing::info!(
            width = actual_width,
            height = actual_height,
            fps = config.frame_rate,
            pixel_format = ?config.pixel_format,
            audio = captures_audio,
            "Screen capture started"
        );

        Ok(Self {
            stream,
            frame_rx,
        })
    }

    /// Receive the next capture event (async, cancellation safe)
    pub async fn next_frame(&mut self) -> Option<CaptureEvent> {
        self.frame_rx.recv().await
    }

    /// Try to get a buffered capture event without waiting. Returns None if no event ready.
    pub fn try_next_frame(&mut self) -> Option<CaptureEvent> {
        self.frame_rx.try_recv().ok()
    }

    /// Update capture frame rate at runtime.
    /// Note: SCStream::update_configuration() blocks until completion via internal semaphore.
    pub fn set_frame_rate(&self, fps: u32) -> anyhow::Result<()> {
        use screencapturekit::cm::CMTime;
        use screencapturekit::prelude::SCStreamConfiguration;

        let interval = CMTime::new(1, fps as i32);
        let mut config = SCStreamConfiguration::new();
        config.set_minimum_frame_interval(&interval);
        self.stream
            .update_configuration(&config)
            .map_err(|e| anyhow::anyhow!("Failed to update SCStream frame rate: {:?}", e))?;
        tracing::info!(fps, "Capture frame rate updated");
        Ok(())
    }
}

/// Fallback capturer using CGDisplayCreateImage (CoreGraphics).
/// Works during lock screen because it captures at the display level,
/// below the window server / ScreenCaptureKit layer.
pub struct CgFallbackCapturer {
    display_id: u32,
    width: u32,
    height: u32,
    frame_interval: std::time::Duration,
}

impl CgFallbackCapturer {
    /// Create a fallback capturer for the main display
    pub fn new(config: &CaptureConfig) -> Self {
        let display_id = core_graphics::display::CGDisplay::main().id;
        let fps = if config.frame_rate > 0 { config.frame_rate } else { 30 };
        Self {
            display_id,
            width: config.width,
            height: config.height,
            frame_interval: std::time::Duration::from_micros(1_000_000 / fps as u64),
        }
    }

    /// Capture a single frame using CGDisplayCreateImage
    pub fn capture_frame(&self) -> Option<CapturedFrame> {
        let display = core_graphics::display::CGDisplay::new(self.display_id);
        let image = display.image()?;

        let w = image.width() as u32;
        let h = image.height() as u32;
        let bpr = image.bytes_per_row();
        let data = image.data();
        let raw = data.bytes().to_vec();

        Some(CapturedFrame {
            width: if self.width > 0 { self.width } else { w },
            height: if self.height > 0 { self.height } else { h },
            data: FrameData::Raw(Bytes::from(raw)),
            stride: bpr,
            timestamp_us: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_micros() as u64,
            dirty_rects: vec![],
        })
    }

    /// Frame interval for pacing
    pub fn frame_interval(&self) -> std::time::Duration {
        self.frame_interval
    }
}

// ---------------------------------------------------------------------------
// CoreVideo FFI for CVPixelBuffer retain/release and plane access
// ---------------------------------------------------------------------------

#[link(name = "CoreVideo", kind = "framework")]
extern "C" {
    fn CVPixelBufferRetain(pixel_buffer: *mut c_void) -> *mut c_void;
    fn CVPixelBufferRelease(pixel_buffer: *mut c_void);
    fn CVPixelBufferLockBaseAddress(pixel_buffer: *mut c_void, flags: u64) -> i32;
    fn CVPixelBufferUnlockBaseAddress(pixel_buffer: *mut c_void, flags: u64) -> i32;
    fn CVPixelBufferGetBaseAddressOfPlane(pixel_buffer: *mut c_void, plane: usize) -> *mut u8;
    fn CVPixelBufferGetBytesPerRowOfPlane(pixel_buffer: *mut c_void, plane: usize) -> usize;
    fn CVPixelBufferGetHeightOfPlane(pixel_buffer: *mut c_void, plane: usize) -> usize;
}

/// kCVPixelBufferLock_ReadOnly
const CV_PIXEL_BUFFER_LOCK_READ_ONLY: u64 = 0x0000_0001;

// ---------------------------------------------------------------------------
// NV12PlaneData — extracted Y and UV plane data from an NV12 pixel buffer
// ---------------------------------------------------------------------------

/// Holds copied plane data from an NV12 CVPixelBuffer.
/// Used for the OpenH264 software encoding fallback path.
pub struct NV12PlaneData {
    /// Y (luma) plane data, one byte per pixel, row-major
    pub y_data: Vec<u8>,
    /// Y plane stride (bytes per row, may include padding)
    pub y_stride: usize,
    /// UV (chroma) plane data, interleaved U/V, half resolution
    pub uv_data: Vec<u8>,
    /// UV plane stride (bytes per row, may include padding)
    pub uv_stride: usize,
    /// Width of the Y plane in pixels
    pub width: usize,
    /// Height of the Y plane in pixels
    pub height: usize,
}

// ---------------------------------------------------------------------------
// SafePixelBuffer — RAII wrapper around a retained CVPixelBufferRef
// ---------------------------------------------------------------------------

/// A safe RAII wrapper around a `CVPixelBufferRef` that manages the
/// retain/release lifecycle. Intended for zero-copy frame passing to
/// VideoToolbox (hardware encoder) while also supporting a locked-read
/// path for OpenH264 (software encoder fallback).
///
/// # Safety
///
/// The inner pointer must originate from a valid `CVPixelBufferRef`.
/// `Send` is implemented because IOSurface-backed pixel buffers are safe
/// to transfer across threads. `Sync` is deliberately NOT implemented
/// because `CVPixelBufferLockBaseAddress` / `UnlockBaseAddress` are not
/// safe for concurrent access from multiple threads.
pub struct SafePixelBuffer {
    ptr: *mut c_void,
}

// SAFETY: IOSurface-backed CVPixelBuffers can be sent across threads.
// We do NOT implement Sync — lock/unlock is not thread-safe for
// concurrent access.
unsafe impl Send for SafePixelBuffer {}

impl SafePixelBuffer {
    /// Create a `SafePixelBuffer` by retaining the given `CVPixelBufferRef`.
    ///
    /// # Safety
    ///
    /// `ptr` must be a valid, non-null `CVPixelBufferRef`.
    pub unsafe fn from_raw(ptr: *mut c_void) -> Self {
        debug_assert!(!ptr.is_null(), "CVPixelBufferRef must not be null");
        CVPixelBufferRetain(ptr);
        Self { ptr }
    }

    /// Return the raw `CVPixelBufferRef` pointer (e.g. for passing to
    /// VideoToolbox's `VTCompressionSessionEncodeFrame`).
    pub fn as_ptr(&self) -> *mut c_void {
        self.ptr
    }

    /// Create a new `SafePixelBuffer` that shares the same underlying
    /// `CVPixelBuffer`, incrementing its retain count.
    pub fn clone_ref(&self) -> Self {
        unsafe {
            CVPixelBufferRetain(self.ptr);
        }
        Self { ptr: self.ptr }
    }

    /// Lock the pixel buffer, copy NV12 plane data out, and unlock.
    ///
    /// This is the software-encoding path: we lock the buffer read-only,
    /// memcpy the Y and UV planes into owned `Vec<u8>`s, then unlock.
    /// The lock is held for the shortest possible duration.
    ///
    /// Returns `None` if the lock fails or plane pointers are null.
    pub fn lock_and_read_nv12(&self) -> Option<NV12PlaneData> {
        unsafe {
            // Lock for read-only access
            let status = CVPixelBufferLockBaseAddress(self.ptr, CV_PIXEL_BUFFER_LOCK_READ_ONLY);
            if status != 0 {
                tracing::warn!(status, "CVPixelBufferLockBaseAddress failed");
                return None;
            }

            let result = self.read_nv12_planes();

            // Always unlock, even if plane read failed
            CVPixelBufferUnlockBaseAddress(self.ptr, CV_PIXEL_BUFFER_LOCK_READ_ONLY);

            result
        }
    }

    /// Read Y and UV planes while the buffer is locked.
    /// Caller must ensure the buffer is locked before calling.
    unsafe fn read_nv12_planes(&self) -> Option<NV12PlaneData> {
        // Plane 0 = Y (luma)
        let y_ptr = CVPixelBufferGetBaseAddressOfPlane(self.ptr, 0);
        let y_stride = CVPixelBufferGetBytesPerRowOfPlane(self.ptr, 0);
        let y_height = CVPixelBufferGetHeightOfPlane(self.ptr, 0);

        // Plane 1 = UV (chroma, interleaved)
        let uv_ptr = CVPixelBufferGetBaseAddressOfPlane(self.ptr, 1);
        let uv_stride = CVPixelBufferGetBytesPerRowOfPlane(self.ptr, 1);
        let uv_height = CVPixelBufferGetHeightOfPlane(self.ptr, 1);

        if y_ptr.is_null() || uv_ptr.is_null() {
            tracing::warn!("NV12 plane base address is null");
            return None;
        }

        let y_len = y_stride * y_height;
        let uv_len = uv_stride * uv_height;

        let y_data = std::slice::from_raw_parts(y_ptr, y_len).to_vec();
        let uv_data = std::slice::from_raw_parts(uv_ptr, uv_len).to_vec();

        // Width is derived from plane 0 stride and pixel format.
        // For NV12 Y plane, each pixel is one byte, but stride may include
        // padding. We use the plane height directly and report stride so
        // callers can handle padding.
        Some(NV12PlaneData {
            y_data,
            y_stride,
            uv_data,
            uv_stride,
            width: y_stride, // conservative: callers should clamp to actual width
            height: y_height,
        })
    }
}

impl Drop for SafePixelBuffer {
    fn drop(&mut self) {
        // SAFETY: ptr was retained in `from_raw`, so we must release it.
        unsafe {
            CVPixelBufferRelease(self.ptr);
        }
    }
}

impl std::fmt::Debug for SafePixelBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SafePixelBuffer")
            .field("ptr", &self.ptr)
            .finish()
    }
}
