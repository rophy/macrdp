//! EGFX frame window: how many frames may wait for a FrameAcknowledge.
//!
//! MS-RDPEGFX flow control uses RDPGFX_FRAME_ACKNOWLEDGE_PDU (3.2.5.13).
//! It has no window size from the client. The core Frame Acknowledge
//! capability (MS-RDPRFX 2.2.1.3, maxUnacknowledgedFrameCount) is for frames
//! of surface commands (TS_FRAME_ACKNOWLEDGE_PDU, MS-RDPRFX 2.2.3.1), not for
//! EGFX. mstsc sends 1 there; used as the EGFX window, it allows one frame per
//! round trip (about 20 fps on a LAN).
//!
//! The window here is the number of frames that fill one base round trip at
//! the capture rate, plus one. The base round trip is the smallest
//! send-to-acknowledge time of the channel: the path and decode time without
//! queueing. It is not a recent minimum: when the client decodes slower than
//! the capture rate (mstsc in a VM shows ~33 fps), every frame waits in the
//! client and a recent minimum grows with the window (measured: 20 -> 120 ms
//! base, window 3 -> 8, ack RTT 170 ms). A larger window then only adds delay.
//! Limit: if the path gets slower during a session, the window stays at the
//! size of the faster path (never below `MIN_WINDOW`) until the next connection.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

/// Smallest window. Also the window until the first acknowledgement.
pub const MIN_WINDOW: u32 = 3;
/// Largest window: 60 fps up to about 115 ms base round trip.
pub const MAX_WINDOW: u32 = 8;
/// Frames older than this without an acknowledgement are forgotten.
const SENT_TIMEOUT: Duration = Duration::from_secs(5);
/// MS-RDPEGFX 2.2.2.13 SUSPEND_FRAME_ACKNOWLEDGEMENT.
const SUSPEND_QUEUE_DEPTH: u32 = 0xFFFF_FFFF;
/// A video frame sent within this time after the previous one is part of a
/// stream. Only those give queueing-delay samples: the first frame after a
/// quiet screen also waits for an idle Wi-Fi link to wake up (50-200 ms).
const STREAM_GAP: Duration = Duration::from_millis(100);
/// Queueing-delay samples are kept for this long.
pub const DELAY_SPAN: Duration = Duration::from_secs(1);

/// What a sent EGFX frame carries.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameKind {
    /// An H.264 frame.
    Video,
    /// Anything else (lossless refinement): large and rare, not a congestion sample.
    Other,
}

#[derive(Debug, Default)]
pub struct FrameWindow {
    /// Frames sent and not acknowledged: (frame id, send time, congestion
    /// sample), oldest first.
    sent: VecDeque<(u32, Instant, bool)>,
    /// When the last video frame was sent.
    last_video: Option<Instant>,
    /// Queueing delays (RTT above the base RTT) of recent stream frames.
    delays: VecDeque<(Instant, Duration)>,
    /// Smallest RTT of the channel.
    base_rtt: Option<Duration>,
    /// Last queueDepth from the client (bytes not yet decoded).
    pub last_queue_depth: u32,
}

impl FrameWindow {
    /// Forget everything (new connection or new channel).
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// A frame with `frame_id` was put on the wire.
    pub fn on_sent(&mut self, frame_id: u32, kind: FrameKind, now: Instant) {
        while self
            .sent
            .front()
            .is_some_and(|&(_, t, _)| now.duration_since(t) > SENT_TIMEOUT)
        {
            self.sent.pop_front();
        }
        let sample = kind == FrameKind::Video
            && self
                .last_video
                .is_some_and(|t| now.duration_since(t) <= STREAM_GAP);
        if kind == FrameKind::Video {
            self.last_video = Some(now);
        }
        self.sent.push_back((frame_id, now, sample));
    }

    /// The client acknowledged `frame_id`. Returns the RTT of that frame.
    pub fn on_ack(&mut self, frame_id: u32, queue_depth: u32, now: Instant) -> Option<Duration> {
        if queue_depth == SUSPEND_QUEUE_DEPTH {
            // No more acknowledgements: the pipeline does not wait for them.
            self.sent.clear();
            return None;
        }
        self.last_queue_depth = queue_depth;
        let pos = self.sent.iter().position(|&(id, _, _)| id == frame_id)?;
        let (_, sent_at, sample) = self.sent[pos];
        // Acknowledgements come in order; older entries were lost or skipped.
        self.sent.drain(..=pos);
        let rtt = now.duration_since(sent_at);
        let base = self.base_rtt.map_or(rtt, |b| b.min(rtt));
        self.base_rtt = Some(base);
        if sample {
            self.delays.push_back((now, rtt - base));
        }
        Some(rtt)
    }

    /// Median queueing delay (RTT above the base RTT) of the stream frames
    /// acknowledged in the last `span` (at most `DELAY_SPAN`), and the number
    /// of samples.
    pub fn queue_delay(&mut self, now: Instant, span: Duration) -> Option<(Duration, usize)> {
        while self
            .delays
            .front()
            .is_some_and(|&(t, _)| now.duration_since(t) > DELAY_SPAN)
        {
            self.delays.pop_front();
        }
        let mut d: Vec<Duration> = self
            .delays
            .iter()
            .filter(|&&(t, _)| now.duration_since(t) <= span)
            .map(|&(_, d)| d)
            .collect();
        if d.is_empty() {
            return None;
        }
        d.sort_unstable();
        Some((d[d.len() / 2], d.len()))
    }

    /// Smallest RTT of the channel.
    pub fn base_rtt(&self) -> Option<Duration> {
        self.base_rtt
    }

    /// Frames that may be in flight at `frame_interval` per frame.
    pub fn target(&self, frame_interval: Duration) -> u32 {
        let Some(base) = self.base_rtt else {
            return MIN_WINDOW;
        };
        if frame_interval.is_zero() {
            return MAX_WINDOW;
        }
        let frames = base.as_secs_f64() / frame_interval.as_secs_f64();
        let window = frames.ceil().min(f64::from(MAX_WINDOW)) as u32 + 1;
        window.clamp(MIN_WINDOW, MAX_WINDOW)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const FRAME_60: Duration = Duration::from_micros(16_667);

    fn ms(v: u64) -> Duration {
        Duration::from_millis(v)
    }

    /// Send a frame and acknowledge it `rtt` later.
    fn round_trip(w: &mut FrameWindow, id: u32, at: Instant, rtt: Duration) {
        w.on_sent(id, FrameKind::Video, at);
        assert_eq!(w.on_ack(id, 0, at + rtt), Some(rtt));
    }

    #[test]
    fn min_window_before_any_ack() {
        let w = FrameWindow::default();
        assert_eq!(w.target(FRAME_60), MIN_WINDOW);
    }

    #[test]
    fn lan_round_trip_keeps_min_window() {
        let mut w = FrameWindow::default();
        let t0 = Instant::now();
        round_trip(&mut w, 0, t0, ms(10));
        assert_eq!(w.target(FRAME_60), MIN_WINDOW);
    }

    #[test]
    fn window_covers_the_round_trip() {
        let mut w = FrameWindow::default();
        let t0 = Instant::now();
        // 60 ms = 3.6 frames at 60 fps -> 4 + 1.
        round_trip(&mut w, 0, t0, ms(60));
        assert_eq!(w.target(FRAME_60), 5);
        // 30 fps: 2 frames + 1.
        assert_eq!(w.target(ms(33)), 3);
        // 110 ms: 7 + 1.
        let mut w2 = FrameWindow::default();
        round_trip(&mut w2, 0, t0, ms(110));
        assert_eq!(w2.target(FRAME_60), 8);
    }

    #[test]
    fn window_is_capped() {
        let mut w = FrameWindow::default();
        let t0 = Instant::now();
        round_trip(&mut w, 0, t0, ms(500));
        assert_eq!(w.target(FRAME_60), MAX_WINDOW);
        assert_eq!(w.target(Duration::ZERO), MAX_WINDOW);
    }

    #[test]
    fn queueing_does_not_grow_the_window() {
        let mut w = FrameWindow::default();
        let t0 = Instant::now();
        round_trip(&mut w, 0, t0, ms(20));
        // Later frames wait in the client: larger RTT, same base.
        for id in 1..20 {
            round_trip(&mut w, id, t0 + ms(u64::from(id) * 17), ms(90));
        }
        assert_eq!(w.base_rtt(), Some(ms(20)));
        assert_eq!(w.target(FRAME_60), MIN_WINDOW);
    }

    #[test]
    fn slow_client_does_not_grow_the_window() {
        let mut w = FrameWindow::default();
        let t0 = Instant::now();
        round_trip(&mut w, 0, t0, ms(20));
        // The client stays behind for a long time: all later RTTs are high.
        for id in 1..2000 {
            round_trip(&mut w, id, t0 + ms(u64::from(id) * 30), ms(120));
        }
        assert_eq!(w.target(FRAME_60), MIN_WINDOW);
    }

    #[test]
    fn reset_forgets_the_base() {
        let mut w = FrameWindow::default();
        let t0 = Instant::now();
        round_trip(&mut w, 0, t0, ms(10));
        w.reset();
        round_trip(&mut w, 1, t0, ms(80));
        assert_eq!(w.target(FRAME_60), 6);
    }

    #[test]
    fn ack_of_later_frame_drops_older_entries() {
        let mut w = FrameWindow::default();
        let t0 = Instant::now();
        w.on_sent(0, FrameKind::Video, t0);
        w.on_sent(1, FrameKind::Video, t0 + ms(16));
        assert_eq!(w.on_ack(1, 0, t0 + ms(40)), Some(ms(24)));
        // Frame 0 is gone: a late ack for it gives no sample.
        assert_eq!(w.on_ack(0, 0, t0 + ms(50)), None);
    }

    #[test]
    fn unknown_ack_is_ignored() {
        let mut w = FrameWindow::default();
        assert_eq!(w.on_ack(42, 0, Instant::now()), None);
        assert_eq!(w.base_rtt(), None);
    }

    #[test]
    fn suspend_clears_sent_frames() {
        let mut w = FrameWindow::default();
        let t0 = Instant::now();
        w.on_sent(0, FrameKind::Video, t0);
        assert_eq!(w.on_ack(0, SUSPEND_QUEUE_DEPTH, t0 + ms(5)), None);
        assert_eq!(w.on_ack(0, 0, t0 + ms(6)), None);
    }

    #[test]
    fn stale_sent_frames_are_dropped() {
        let mut w = FrameWindow::default();
        let t0 = Instant::now();
        w.on_sent(0, FrameKind::Video, t0);
        w.on_sent(1, FrameKind::Video, t0 + SENT_TIMEOUT + ms(1));
        assert_eq!(w.on_ack(0, 0, t0 + SENT_TIMEOUT + ms(2)), None);
    }

    #[test]
    fn queue_delay_from_stream_frames_only() {
        let mut w = FrameWindow::default();
        let t0 = Instant::now();
        // First frame: no stream yet, but it sets the base (20 ms).
        round_trip(&mut w, 0, t0, ms(20));
        assert_eq!(w.queue_delay(t0 + ms(20), DELAY_SPAN), None);
        // Stream frames 16 ms apart, 50 ms RTT: 30 ms queueing.
        for id in 1..=5 {
            round_trip(&mut w, id, t0 + ms(u64::from(id) * 16), ms(50));
        }
        assert_eq!(w.queue_delay(t0 + ms(140), DELAY_SPAN), Some((ms(30), 5)));
        // A frame after a quiet gap (Wi-Fi wake-up) and a refinement frame give no sample.
        let t1 = t0 + ms(300);
        round_trip(&mut w, 6, t1, ms(300));
        w.on_sent(7, FrameKind::Other, t1 + ms(10));
        w.on_ack(7, 0, t1 + ms(400));
        assert_eq!(
            w.queue_delay(t1 + ms(400), DELAY_SPAN).map(|(_, n)| n),
            Some(5)
        );
        // A shorter span sees only the newer samples.
        assert_eq!(w.queue_delay(t1 + ms(400), ms(10)), None);
        // Samples older than one second are dropped.
        assert_eq!(w.queue_delay(t0 + ms(2000), DELAY_SPAN), None);
    }

    #[test]
    fn queue_depth_is_kept() {
        let mut w = FrameWindow::default();
        let t0 = Instant::now();
        w.on_sent(0, FrameKind::Video, t0);
        w.on_ack(0, 13936, t0 + ms(5));
        assert_eq!(w.last_queue_depth, 13936);
    }
}
