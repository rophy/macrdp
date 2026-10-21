//! EGFX frame window: how many frames may wait for a FrameAcknowledge.
//!
//! Ported from macrdp-v1 (ffb464e, 677670f, bc65e87), relicensed by the author.
//! Here it bounds frames in flight at capture time (`Gfx::submit_bgra`): an
//! encoded frame is never dropped (that would break the H.264 reference chain).
//!
//! MS-RDPEGFX flow control uses RDPGFX_FRAME_ACKNOWLEDGE_PDU (3.2.5.13).
//! It has no window size from the client. The core Frame Acknowledge
//! capability (MS-RDPRFX 2.2.1.3, maxUnacknowledgedFrameCount) is for frames
//! of surface commands (TS_FRAME_ACKNOWLEDGE_PDU, MS-RDPRFX 2.2.3.1), not for
//! EGFX. mstsc sends 1 there; used as the EGFX window, it allows one frame per
//! round trip (about 20 fps on a LAN).
//!
//! The window here is the number of frames that fill one base round trip at
//! the capture rate, plus one. The base round trip is the send-to-acknowledge
//! time of the path and decode without queueing. It is not a recent minimum:
//! when the client decodes slower than the capture rate (mstsc in a VM shows
//! ~33 fps), every frame waits in the client and a recent minimum grows with
//! the window (measured: 20 -> 120 ms base, window 3 -> 8, ack RTT 170 ms). A
//! larger window then only adds delay.
//!
//! A lower RTT lowers the base at once. A higher one can be a slower path or
//! queueing in the client; only a probe tells them apart. When all stream
//! frames of the last second came back `PROBE_MARGIN` above the base, the
//! window drops to 1 for `PROBE_SAMPLES` frames: each is sent when no other
//! frame is in flight, so nothing waits before it in the client, and the
//! smallest of their RTTs is the base RTT of the path now (lower or higher).
//! One sample is not enough: mstsc acknowledges a frame sent alone after 15
//! to 79 ms (measured). A probe that finds no slower path doubles the time
//! to the next one (a slow client keeps all RTTs high and would probe often).

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
/// A probe starts when all stream RTTs of the last `DELAY_SPAN` are this much above the base.
const PROBE_MARGIN: Duration = Duration::from_millis(15);
/// ... and there are at least this many of them.
const PROBE_MIN_SAMPLES: usize = 10;
/// Frames sent alone per probe.
const PROBE_SAMPLES: u8 = 3;
/// Shortest time from one probe to the next ...
const PROBE_INTERVAL_MIN: Duration = Duration::from_secs(4);
/// ... and longest, after probes that found no slower path.
const PROBE_INTERVAL_MAX: Duration = Duration::from_secs(32);
/// A probe that takes longer than this (no frames, lost acknowledgements) ends.
const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// Base RTT probe state.
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
enum Probe {
    #[default]
    Idle,
    /// Window 1. `pending` is the frame sent alone and not acknowledged yet;
    /// `best` the smallest RTT of the `got` frames measured.
    Running {
        since: Instant,
        pending: Option<u32>,
        got: u8,
        best: Option<Duration>,
    },
}

#[derive(Debug, Default)]
pub struct FrameWindow {
    /// Frames sent and not acknowledged: (frame id, send time, stream
    /// frame), oldest first.
    sent: VecDeque<(u32, Instant, bool)>,
    /// When the last video frame was sent.
    last_video: Option<Instant>,
    /// Queueing delays (RTT above the base RTT) of recent stream frames.
    delays: VecDeque<(Instant, Duration)>,
    /// RTTs of recent stream frames (for the probe trigger).
    rtts: VecDeque<(Instant, Duration)>,
    /// RTT of the path without queueing: the last probe or any lower RTT since.
    base_rtt: Option<Duration>,
    probe: Probe,
    last_probe: Option<Instant>,
    /// Time from the last probe to the next one (0 = `PROBE_INTERVAL_MIN`).
    probe_interval: Duration,
}

impl FrameWindow {
    /// Forget everything (new connection or new channel). Here each connection
    /// gets a new window, so only the tests use it.
    #[cfg(test)]
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    /// A frame with `frame_id` was put on the wire.
    pub fn on_sent(&mut self, frame_id: u32, now: Instant) {
        while self
            .sent
            .front()
            .is_some_and(|&(_, t, _)| now.duration_since(t) > SENT_TIMEOUT)
        {
            self.sent.pop_front();
        }
        // A frame sent while others are still in flight is part of a stream
        // too: on a choked link the window spaces frames more than
        // `STREAM_GAP` apart, and without samples the bitrate never steps down.
        let sample = !self.sent.is_empty()
            || self
                .last_video
                .is_some_and(|t| now.duration_since(t) <= STREAM_GAP);
        self.last_video = Some(now);
        if let Probe::Running { pending, .. } = &mut self.probe {
            if pending.is_none() && self.sent.is_empty() {
                *pending = Some(frame_id);
            }
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
        let pos = self.sent.iter().position(|&(id, _, _)| id == frame_id)?;
        let (_, sent_at, sample) = self.sent[pos];
        // Acknowledgements come in order; older entries were lost or skipped.
        self.sent.drain(..=pos);
        let rtt = now.duration_since(sent_at);
        let base = self.base_rtt.map_or(rtt, |b| b.min(rtt));
        self.base_rtt = Some(base);
        self.probe_ack(frame_id, rtt, now);
        let base = self.base_rtt.unwrap_or(rtt);
        if sample {
            self.delays.push_back((now, rtt.saturating_sub(base)));
            self.rtts.push_back((now, rtt));
        }
        self.maybe_start_probe(now);
        Some(rtt)
    }

    /// A frame was acknowledged during a probe.
    fn probe_ack(&mut self, frame_id: u32, rtt: Duration, now: Instant) {
        let Probe::Running {
            since,
            pending: Some(id),
            got,
            best,
        } = self.probe
        else {
            return;
        };
        if id == frame_id {
            let best = best.map_or(rtt, |b| b.min(rtt));
            if got + 1 >= PROBE_SAMPLES {
                self.finish_probe(best, now);
            } else {
                self.probe = Probe::Running {
                    since,
                    pending: None,
                    got: got + 1,
                    best: Some(best),
                };
            }
        } else if !self.sent.iter().any(|&(i, _, _)| i == id) {
            // The probe frame was skipped: send another one alone.
            self.probe = Probe::Running {
                since,
                pending: None,
                got,
                best,
            };
        }
    }

    /// End a probe: its smallest RTT is the base now.
    fn finish_probe(&mut self, best: Duration, now: Instant) {
        let old = self.base_rtt;
        let slower = old.is_none_or(|b| best > b + PROBE_MARGIN);
        self.probe_interval = if slower {
            PROBE_INTERVAL_MIN
        } else {
            (self.probe_interval.max(PROBE_INTERVAL_MIN) * 2).min(PROBE_INTERVAL_MAX)
        };
        tracing::debug!(
            rtt_ms = best.as_millis(),
            old_ms = ?old.map(|b| b.as_millis()),
            next_s = self.probe_interval.as_secs(),
            "EGFX base RTT probed"
        );
        self.base_rtt = Some(best);
        self.probe = Probe::Idle;
        self.last_probe = Some(now);
    }

    /// Start a probe when every recent stream frame came back well above the base.
    fn maybe_start_probe(&mut self, now: Instant) {
        while self
            .rtts
            .front()
            .is_some_and(|&(t, _)| now.duration_since(t) > DELAY_SPAN)
        {
            self.rtts.pop_front();
        }
        if self.probe != Probe::Idle
            || self.last_probe.is_some_and(|t| {
                now.duration_since(t) < self.probe_interval.max(PROBE_INTERVAL_MIN)
            })
            || self.rtts.len() < PROBE_MIN_SAMPLES
        {
            return;
        }
        let (Some(base), Some(lowest)) = (self.base_rtt, self.rtts.iter().map(|&(_, r)| r).min())
        else {
            return;
        };
        if lowest > base + PROBE_MARGIN {
            self.probe = Probe::Running {
                since: now,
                pending: None,
                got: 0,
                best: None,
            };
            self.rtts.clear();
        }
    }

    /// End a probe that takes too long (quiet screen, lost acknowledgements).
    fn check_probe_timeout(&mut self, now: Instant) {
        if let Probe::Running { since, best, .. } = self.probe {
            if now.duration_since(since) > PROBE_TIMEOUT {
                match best {
                    Some(best) => self.finish_probe(best, now),
                    None => {
                        self.probe = Probe::Idle;
                        self.last_probe = Some(now);
                    }
                }
            }
        }
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
    #[cfg(test)]
    pub fn base_rtt(&self) -> Option<Duration> {
        self.base_rtt
    }

    /// Frames sent and not acknowledged (or forgotten) yet.
    pub fn in_flight(&self) -> u32 {
        u32::try_from(self.sent.len()).unwrap_or(u32::MAX)
    }

    /// Frames that may be in flight at `frame_interval` per frame.
    pub fn target(&mut self, frame_interval: Duration, now: Instant) -> u32 {
        self.check_probe_timeout(now);
        if matches!(self.probe, Probe::Running { .. }) {
            return 1;
        }
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
        w.on_sent(id, at);
        assert_eq!(w.on_ack(id, 0, at + rtt), Some(rtt));
    }

    #[test]
    fn min_window_before_any_ack() {
        let mut w = FrameWindow::default();
        assert_eq!(w.target(FRAME_60, Instant::now()), MIN_WINDOW);
    }

    #[test]
    fn lan_round_trip_keeps_min_window() {
        let mut w = FrameWindow::default();
        let t0 = Instant::now();
        round_trip(&mut w, 0, t0, ms(10));
        assert_eq!(w.target(FRAME_60, Instant::now()), MIN_WINDOW);
    }

    #[test]
    fn window_covers_the_round_trip() {
        let mut w = FrameWindow::default();
        let t0 = Instant::now();
        // 60 ms = 3.6 frames at 60 fps -> 4 + 1.
        round_trip(&mut w, 0, t0, ms(60));
        assert_eq!(w.target(FRAME_60, Instant::now()), 5);
        // 30 fps: 2 frames + 1.
        assert_eq!(w.target(ms(33), Instant::now()), 3);
        // 110 ms: 7 + 1.
        let mut w2 = FrameWindow::default();
        round_trip(&mut w2, 0, t0, ms(110));
        assert_eq!(w2.target(FRAME_60, Instant::now()), 8);
    }

    #[test]
    fn window_is_capped() {
        let mut w = FrameWindow::default();
        let t0 = Instant::now();
        round_trip(&mut w, 0, t0, ms(500));
        assert_eq!(w.target(FRAME_60, Instant::now()), MAX_WINDOW);
        assert_eq!(w.target(Duration::ZERO, Instant::now()), MAX_WINDOW);
    }

    /// Server at 60 fps limited by the window; one-way delay `path(t)` each
    /// way; the client decodes one frame at a time in `decode`. Runs from
    /// `from` to `to` (ms since `t0`) and returns the windows used.
    struct Sim {
        w: FrameWindow,
        t0: Instant,
        next_id: u32,
        /// (frame id, ack arrival time) in order
        acks: VecDeque<(u32, Instant)>,
        client_free: Instant,
    }

    impl Sim {
        fn new() -> Self {
            let t0 = Instant::now();
            Self {
                w: FrameWindow::default(),
                t0,
                next_id: 0,
                acks: VecDeque::new(),
                client_free: t0,
            }
        }

        fn run(&mut self, from_ms: u64, to_ms: u64, path: Duration, decode: Duration) -> Vec<u32> {
            let mut windows = Vec::new();
            let mut tick = Duration::from_millis(from_ms);
            while tick < Duration::from_millis(to_ms) {
                let now = self.t0 + tick;
                while self.acks.front().is_some_and(|&(_, at)| at <= now) {
                    let (id, at) = self.acks.pop_front().unwrap();
                    self.w.on_ack(id, 0, at);
                }
                let window = self.w.target(FRAME_60, now);
                windows.push(window);
                let in_flight =
                    self.next_id - self.acks.front().map_or(self.next_id, |&(id, _)| id);
                if in_flight < window {
                    let id = self.next_id;
                    self.next_id += 1;
                    self.w.on_sent(id, now);
                    let start = (now + path).max(self.client_free);
                    self.client_free = start + decode;
                    // TCP keeps the order of the acknowledgements.
                    let ack =
                        (self.client_free + path).max(self.acks.back().map_or(now, |&(_, a)| a));
                    self.acks.push_back((id, ack));
                }
                tick += FRAME_60;
            }
            windows
        }
    }

    fn last_second(windows: &[u32]) -> &[u32] {
        &windows[windows.len() - 60..]
    }

    #[test]
    fn queueing_does_not_grow_the_window() {
        // 2 ms each way, the client decodes 25 ms per frame (40 fps): frames queue in the client.
        let mut sim = Sim::new();
        let windows = sim.run(0, 30_000, ms(2), ms(25));
        assert!(
            windows[600..].iter().all(|&w| w <= MIN_WINDOW),
            "{windows:?}"
        );
        assert_eq!(sim.w.base_rtt(), Some(ms(29)));
    }

    #[test]
    fn slower_path_grows_the_window_and_faster_path_shrinks_it() {
        let mut sim = Sim::new();
        let fast = sim.run(0, 5_000, ms(5), ms(2));
        assert!(last_second(&fast).iter().all(|&w| w == MIN_WINDOW));
        // 50 ms more each way (+100 ms RTT) in the middle of the session.
        let slow = sim.run(5_000, 15_000, ms(55), ms(2));
        assert_eq!(*slow.last().unwrap(), MAX_WINDOW, "{slow:?}");
        assert!(last_second(&slow).iter().all(|&w| w == MAX_WINDOW));
        // Back to the fast path: the window goes down within a round trip.
        let back = sim.run(15_000, 17_000, ms(5), ms(2));
        assert!(
            last_second(&back).iter().all(|&w| w == MIN_WINDOW),
            "{back:?}"
        );
    }

    #[test]
    fn probes_back_off_when_the_path_is_the_same() {
        // A slow client keeps the RTT high: probes at 4, 8, 16, 32, 32 s intervals.
        let mut sim = Sim::new();
        let windows = sim.run(0, 120_000, ms(2), ms(25));
        let probes = windows
            .windows(2)
            .filter(|p| p[0] != 1 && p[1] == 1)
            .count();
        assert!((4..=6).contains(&probes), "{probes} probes");
        assert_eq!(sim.w.probe_interval, PROBE_INTERVAL_MAX);
    }

    #[test]
    fn probe_takes_the_smallest_of_its_samples() {
        let mut w = FrameWindow::default();
        let t0 = Instant::now();
        w.base_rtt = Some(ms(10));
        w.probe = Probe::Running {
            since: t0,
            pending: None,
            got: 0,
            best: None,
        };
        let mut t = t0;
        for (id, rtt) in [(1, 70), (2, 40), (3, 55)] {
            assert_eq!(w.target(FRAME_60, t), 1);
            round_trip(&mut w, id, t, ms(rtt));
            t += ms(rtt + 1);
        }
        assert_eq!(w.base_rtt(), Some(ms(40)));
        assert_eq!(w.probe, Probe::Idle);
        assert_eq!(w.probe_interval, PROBE_INTERVAL_MIN);
    }

    #[test]
    fn probe_without_acks_is_given_up() {
        let mut w = FrameWindow::default();
        let t0 = Instant::now();
        w.base_rtt = Some(ms(10));
        w.probe = Probe::Running {
            since: t0,
            pending: None,
            got: 0,
            best: None,
        };
        assert_eq!(w.target(FRAME_60, t0 + ms(10)), 1);
        assert_eq!(w.target(FRAME_60, t0 + PROBE_TIMEOUT + ms(1)), MIN_WINDOW);
    }

    #[test]
    fn reset_forgets_the_base() {
        let mut w = FrameWindow::default();
        let t0 = Instant::now();
        round_trip(&mut w, 0, t0, ms(10));
        w.reset();
        round_trip(&mut w, 1, t0, ms(80));
        assert_eq!(w.target(FRAME_60, Instant::now()), 6);
    }

    #[test]
    fn ack_of_later_frame_drops_older_entries() {
        let mut w = FrameWindow::default();
        let t0 = Instant::now();
        w.on_sent(0, t0);
        w.on_sent(1, t0 + ms(16));
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
        w.on_sent(0, t0);
        assert_eq!(w.on_ack(0, SUSPEND_QUEUE_DEPTH, t0 + ms(5)), None);
        assert_eq!(w.on_ack(0, 0, t0 + ms(6)), None);
    }

    #[test]
    fn stale_sent_frames_are_dropped() {
        let mut w = FrameWindow::default();
        let t0 = Instant::now();
        w.on_sent(0, t0);
        w.on_sent(1, t0 + SENT_TIMEOUT + ms(1));
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
        // A frame after a quiet gap (Wi-Fi wake-up) gives no sample.
        let t1 = t0 + ms(300);
        round_trip(&mut w, 6, t1, ms(300));
        assert_eq!(
            w.queue_delay(t1 + ms(300), DELAY_SPAN).map(|(_, n)| n),
            Some(5)
        );
        // A shorter span sees only the newer samples.
        assert_eq!(w.queue_delay(t1 + ms(300), ms(10)), None);
        // Samples older than one second are dropped.
        assert_eq!(w.queue_delay(t0 + ms(2000), DELAY_SPAN), None);
    }

    #[test]
    fn frames_spaced_by_a_full_window_still_give_samples() {
        let mut w = FrameWindow::default();
        let t0 = Instant::now();
        round_trip(&mut w, 0, t0, ms(20));
        // A choked link: 3 frames in flight, each acked 400 ms after it left,
        // so frames leave 150 ms apart (more than STREAM_GAP).
        // Frame 1 follows a quiet gap: no sample. Frames 2 and 3 leave behind it.
        let mut t = t0 + ms(200);
        for id in 1..=3 {
            w.on_sent(id, t);
            t += ms(150);
        }
        w.on_ack(1, 0, t0 + ms(600));
        assert_eq!(w.queue_delay(t0 + ms(600), DELAY_SPAN), None);
        w.on_ack(2, 0, t0 + ms(750));
        let (d, n) = w.queue_delay(t0 + ms(750), DELAY_SPAN).unwrap();
        assert_eq!(n, 1);
        assert_eq!(d, ms(380));
    }

    #[test]
    fn in_flight_counts_unacknowledged_frames() {
        let mut w = FrameWindow::default();
        let t0 = Instant::now();
        w.on_sent(0, t0);
        w.on_sent(1, t0 + ms(16));
        w.on_sent(2, t0 + ms(33));
        assert_eq!(w.in_flight(), 3);
        w.on_ack(1, 0, t0 + ms(40));
        assert_eq!(w.in_flight(), 1);
        w.on_ack(2, SUSPEND_QUEUE_DEPTH, t0 + ms(50));
        assert_eq!(w.in_flight(), 0);
    }
}
