//! Keep the Mac's Wi-Fi radio awake while a client uses the session.
//!
//! With Wi-Fi power save, macOS lets the radio sleep when the Mac has sent
//! nothing for a short time (measured: one packet each 200 ms keeps it awake,
//! one each 330 ms does not). The access point then holds packets for the Mac
//! until the radio wakes up: a ping to an idle Mac took 70 ms on average and
//! up to 180 ms, against 5 ms while the Mac sends. So a key press after a
//! quiet moment reached the server 50-100 ms late (mstsc key-to-screen
//! 155 ms on an idle link, 105 ms on a busy one).
//!
//! While a client shows the session and sent input not long ago, the display
//! loop sends a small ECHO request (MS-RDPEECO, 1 byte of payload; the client
//! answers it) when nothing else went out for `INTERVAL`.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use ironrdp_server::EchoServerHandle;

/// Longest time without sending (the radio stayed awake at 200 ms).
pub const INTERVAL: Duration = Duration::from_millis(100);
/// Keep the radio awake for this long after the last client input.
const AFTER_INPUT: Duration = Duration::from_secs(600);
/// Requests without an answer before the loop stops sending (the client did
/// not open the ECHO channel).
const MAX_UNANSWERED: u32 = 5;

/// Shared between the server (ECHO handle), the input handler and the display loop.
#[derive(Clone)]
pub struct KeepAwake {
    echo: Arc<OnceLock<EchoServerHandle>>,
    /// Time of the last client input, in ms since `epoch` (0: none yet)
    last_input_ms: Arc<AtomicU64>,
    epoch: Instant,
}

impl KeepAwake {
    pub fn new() -> Self {
        Self {
            echo: Arc::new(OnceLock::new()),
            last_input_ms: Arc::new(AtomicU64::new(0)),
            epoch: Instant::now(),
        }
    }

    /// Set once, after the server is built.
    pub fn set_echo(&self, handle: EchoServerHandle) {
        let _ = self.echo.set(handle);
    }

    /// The client sent input.
    pub fn note_input(&self) {
        let ms = self.epoch.elapsed().as_millis() as u64;
        self.last_input_ms.store(ms.max(1), Ordering::Relaxed);
    }

    fn input_recent(&self, now: Instant) -> bool {
        match self.last_input_ms.load(Ordering::Relaxed) {
            0 => false,
            ms => {
                now.saturating_duration_since(self.epoch + Duration::from_millis(ms)) < AFTER_INPUT
            }
        }
    }
}

/// Per display loop (one connection).
pub struct KeepAwakeLoop {
    shared: KeepAwake,
    last_output: Instant,
    unanswered: u32,
    stopped: bool,
}

impl KeepAwakeLoop {
    pub fn new(shared: KeepAwake) -> Self {
        Self {
            shared,
            last_output: Instant::now(),
            unanswered: 0,
            stopped: false,
        }
    }

    /// Something went out to the client.
    pub fn on_output(&mut self, now: Instant) {
        self.last_output = now;
    }

    /// When to check again; None while there is nothing to keep awake.
    pub fn due(&self, now: Instant) -> Option<Instant> {
        (!self.stopped && self.shared.input_recent(now)).then_some(self.last_output + INTERVAL)
    }

    /// Send a request if nothing went out for `INTERVAL`.
    pub fn tick(&mut self, now: Instant) {
        if self.stopped || !self.shared.input_recent(now) || now < self.last_output + INTERVAL {
            return;
        }
        self.last_output = now;
        let Some(echo) = self.shared.echo.get().cloned() else {
            return;
        };
        if self.count_answer(echo.take_measurements().len()) {
            tracing::info!(
                requests = MAX_UNANSWERED,
                "Client does not answer ECHO requests: Wi-Fi keep-awake stopped"
            );
            return;
        }
        if let Err(e) = echo.send_request(vec![0x6b]) {
            tracing::debug!("Keep-awake ECHO request not sent: {e}");
        }
    }

    /// Count one request against the answers that came back since the last
    /// one. Returns true when the loop stops sending.
    fn count_answer(&mut self, answers: usize) -> bool {
        if answers > 0 {
            self.unanswered = 0;
        }
        self.unanswered += 1;
        if self.unanswered > MAX_UNANSWERED {
            self.stopped = true;
        }
        self.stopped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn looped(input: bool) -> (KeepAwakeLoop, Instant) {
        let shared = KeepAwake::new();
        if input {
            shared.note_input();
        }
        let now = Instant::now();
        let mut l = KeepAwakeLoop::new(shared);
        l.on_output(now);
        (l, now)
    }

    #[test]
    fn nothing_due_without_input() {
        let (l, now) = looped(false);
        assert_eq!(l.due(now), None);
    }

    #[test]
    fn due_one_interval_after_the_last_output() {
        let (mut l, now) = looped(true);
        assert_eq!(l.due(now), Some(now + INTERVAL));
        l.on_output(now + Duration::from_millis(30));
        assert_eq!(l.due(now), Some(now + Duration::from_millis(30) + INTERVAL));
    }

    #[test]
    fn nothing_due_long_after_the_last_input() {
        let (l, now) = looped(true);
        assert!(l.due(now + AFTER_INPUT - Duration::from_secs(1)).is_some());
        assert_eq!(l.due(now + AFTER_INPUT + Duration::from_secs(1)), None);
    }

    #[test]
    fn stops_when_the_client_does_not_answer() {
        let (mut l, now) = looped(true);
        for _ in 0..MAX_UNANSWERED {
            assert!(!l.count_answer(0));
        }
        assert!(l.count_answer(0));
        assert_eq!(l.due(now), None);
    }

    #[test]
    fn answers_keep_it_going() {
        let (mut l, _) = looped(true);
        for _ in 0..3 * MAX_UNANSWERED {
            assert!(!l.count_answer(1));
        }
        // An answer comes back one request later at most.
        for _ in 0..3 * MAX_UNANSWERED {
            assert!(!l.count_answer(0));
            assert!(!l.count_answer(1));
        }
    }
}
