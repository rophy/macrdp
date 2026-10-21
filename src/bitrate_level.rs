//! Bitrate level for the H.264 stream over TCP: steps the encoder bitrate down
//! when the EGFX frames queue (median queueing delay from
//! `crate::frame_window::FrameWindow::queue_delay`) and back up when the queue
//! stays short.
//!
//! Ported from macrdp-v1 (94168ac, bc65e87, b6e314a, 8f1f45f), relicensed by the
//! author. Single-spike safe: it acts on the median of the last 300 ms (5+
//! samples), not on one frame's round trip.

use std::time::Instant;

/// Encoder bitrate share of each level: full rate, then 80 %, 50 %, 30 %, 15 %.
const LEVEL_QUALITY: [f64; 5] = [1.0, 0.8, 0.5, 0.3, 0.15];
/// Step down when the median queueing delay of the last `DOWN_SPAN` is above
/// this: the link is full. A client that decodes slower than the capture rate
/// queues frames too (mstsc in a VM, ~33 fps: 40-50 ms with a 3-frame window),
/// but a lower bitrate does not help it (measured: same fps at 13 and 44 Mbps).
const QUEUE_DELAY_HIGH_MS: f32 = 100.0;
pub const DOWN_SPAN: std::time::Duration = std::time::Duration::from_millis(300);
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
pub struct BitrateLevel {
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
    /// Start at the 30 % level: a link whose RTT at connect was already high
    /// (the RTT seed in `h264.rs`). It steps up one level per up dwell when the
    /// queue stays short.
    pub fn seeded() -> Self {
        Self {
            level: 3,
            ..Self::default()
        }
    }

    /// Feed the median queueing delay (ms) and sample count of the last
    /// `DOWN_SPAN` (`short`) and of the last second (`long`); returns the
    /// bitrate share (0..=1) to use. Too few samples (a quiet screen): no
    /// change, except one step up for each `QUIET_STEP_UP` of quiet.
    pub fn update(
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

#[cfg(test)]
mod tests {
    use super::*;

    fn run(level: &mut BitrateLevel, t0: Instant, ms: u64, delay: Option<f32>) -> f64 {
        let d = delay.map(|d| (d, MIN_UP_SAMPLES));
        level.update(d, d, t0 + std::time::Duration::from_millis(ms))
    }

    #[test]
    fn seeded_level_starts_low_and_climbs() {
        let t0 = Instant::now();
        let mut l = BitrateLevel::seeded();
        assert_eq!(run(&mut l, t0, 0, Some(10.0)), 0.3);
        assert_eq!(run(&mut l, t0, 3_000, Some(10.0)), 0.5);
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
}
