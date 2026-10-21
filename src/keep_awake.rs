//! Keep the Mac's Wi-Fi radio awake while a client uses the session.
//!
//! Ported from macrdp-v1 (crates/macrdp-server/src/keep_awake.rs, commits
//! 713676d, 92a2dca, 9faf70d, 584b173, 3ff66fb), relicensed to MIT OR
//! Apache-2.0 by the author. Changed here: a timer task reads the time of the
//! last write to the client from the server (`OutputActivity`) instead of
//! being driven by the display loop, and a connection end stops it.
//!
//! With Wi-Fi power save, macOS lets the radio sleep when the Mac has sent
//! nothing for a short time (measured: one packet each 200 ms keeps it awake,
//! one each 330 ms does not). The access point then holds packets for the Mac
//! until the radio wakes up: a ping to an idle Mac took 70 ms on average and
//! up to 180 ms, against 5 ms while the Mac sends. So a key press after a
//! quiet moment reaches the screen 50-150 ms late (key-to-screen p50 on an
//! idle link minus a busy one: mstsc 50-65 ms here, 58 ms with v1; FreeRDP
//! 50-150 ms with v1).
//!
//! While a client is connected and sent input not long ago, the task sends a
//! small ECHO request (MS-RDPEECO, 1 byte of payload; the client answers it)
//! when nothing else went out for `INTERVAL`. A client without the ECHO
//! channel (FreeRDP without `/echo`) gets a Server Heartbeat PDU
//! (MS-RDPBCGR 2.2.16.1, no answer) instead, if it advertised heartbeats.
//! ECHO requests go out only while the client has the ECHO channel open;
//! before it opens it (or when it declines it) a client that takes heartbeats
//! gets Heartbeat PDUs. That choice holds for the connection; a new
//! connection tries ECHO again.

use std::sync::atomic::{AtomicBool, AtomicU64, AtomicU8, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use ironrdp_server::{EchoServerHandle, HeartbeatConfig, HeartbeatHandle, OutputActivity};

/// Longest time without sending (the radio stayed awake at 200 ms).
pub const INTERVAL: Duration = Duration::from_millis(100);
/// Keep the radio awake for this long after the last client input.
const AFTER_INPUT: Duration = Duration::from_secs(600);
/// Requests without an answer before the task stops sending ECHO requests
/// (the client did not open the ECHO channel).
const MAX_UNANSWERED: u32 = 5;
/// How often the task looks again while there is nothing to keep awake.
const IDLE_POLL: Duration = Duration::from_millis(250);
/// Values of the keep-awake heartbeats. They come more often than the period
/// says; the largest period and counts keep a client that watches heartbeats
/// from warning when they stop (after the last input).
const HEARTBEAT: HeartbeatConfig = HeartbeatConfig {
    period_secs: u8::MAX,
    warning_count: u8::MAX,
    reconnect_count: u8::MAX,
};

/// Shared between the server (ECHO and heartbeat handles), the input handler,
/// the connection handler and the timer task.
#[derive(Clone)]
pub struct KeepAwake {
    echo: Arc<OnceLock<EchoServerHandle>>,
    /// On-demand Heartbeat PDUs
    heartbeat: Arc<OnceLock<HeartbeatHandle>>,
    /// The current client takes Heartbeat PDUs
    client_heartbeat: Arc<AtomicBool>,
    /// The current client got a Heartbeat PDU because its ECHO channel was not open
    heartbeat_without_echo: Arc<AtomicBool>,
    /// What keeps the current client's link awake (`Mode` as u8)
    mode: Arc<AtomicU8>,
    /// Counts connections, so the task starts each one afresh
    connection: Arc<AtomicU64>,
    /// Time of the last client input, in ms since `epoch` (0: none, or the
    /// connection ended)
    last_input_ms: Arc<AtomicU64>,
    epoch: Instant,
}

impl KeepAwake {
    pub fn new() -> Self {
        Self {
            echo: Arc::new(OnceLock::new()),
            heartbeat: Arc::new(OnceLock::new()),
            client_heartbeat: Arc::new(AtomicBool::new(false)),
            heartbeat_without_echo: Arc::new(AtomicBool::new(false)),
            mode: Arc::new(AtomicU8::new(Mode::Echo as u8)),
            connection: Arc::new(AtomicU64::new(0)),
            last_input_ms: Arc::new(AtomicU64::new(0)),
            epoch: Instant::now(),
        }
    }

    /// Set once, after the server is built.
    pub fn set_echo(&self, handle: EchoServerHandle) {
        let _ = self.echo.set(handle);
    }

    /// Set once, after the server is built.
    pub fn set_heartbeat(&self, handle: HeartbeatHandle) {
        let _ = self.heartbeat.set(handle);
    }

    /// A new client connected; `heartbeat`: it takes Heartbeat PDUs.
    /// It gets ECHO requests first.
    pub fn on_connect(&self, heartbeat: bool) {
        self.client_heartbeat.store(heartbeat, Ordering::Relaxed);
        self.heartbeat_without_echo.store(false, Ordering::Relaxed);
        self.set_mode(Mode::Echo);
        self.connection.fetch_add(1, Ordering::Relaxed);
    }

    /// The connection ended: nothing to keep awake until the next client
    /// sends input (a request now would wait in the server's queue).
    pub fn on_disconnect(&self) {
        self.last_input_ms.store(0, Ordering::Relaxed);
    }

    /// A Heartbeat PDU goes out because the ECHO channel is not open. True
    /// the first time for the current connection.
    fn first_heartbeat_without_echo(&self) -> bool {
        !self.heartbeat_without_echo.swap(true, Ordering::Relaxed)
    }

    /// The current client has the ECHO channel open.
    fn echo_open(&self) -> bool {
        self.echo.get().is_some_and(|e| e.is_channel_open())
    }

    fn mode(&self) -> Mode {
        match self.mode.load(Ordering::Relaxed) {
            x if x == Mode::Heartbeat as u8 => Mode::Heartbeat,
            x if x == Mode::Stopped as u8 => Mode::Stopped,
            _ => Mode::Echo,
        }
    }

    fn set_mode(&self, mode: Mode) {
        self.mode.store(mode as u8, Ordering::Relaxed);
    }

    /// Ask the server for a Heartbeat PDU. False: no handle set (nothing asked).
    fn send_heartbeat(&self) -> bool {
        match self.heartbeat.get() {
            Some(heartbeat) => {
                heartbeat.send(HEARTBEAT);
                true
            }
            None => false,
        }
    }

    /// The client sent input.
    pub fn note_input(&self) {
        self.note_input_at(Instant::now());
    }

    fn note_input_at(&self, now: Instant) {
        let ms = now.saturating_duration_since(self.epoch).as_millis() as u64;
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

    /// Timer task: sends the keep-awake requests. `output` is the server's
    /// time of the last write to the client.
    pub async fn run(self, output: OutputActivity) {
        let mut l = KeepAwakeLoop::new(self, Instant::now());
        loop {
            if let Some(at) = output.last_write() {
                l.on_output(at);
            }
            let now = Instant::now();
            l.tick(now);
            let wait = match l.due(now) {
                Some(at) => at
                    .saturating_duration_since(Instant::now())
                    .max(Duration::from_millis(5)),
                None => IDLE_POLL,
            };
            tokio::time::sleep(wait).await;
        }
    }
}

/// What a tick sends.
#[derive(Debug, PartialEq, Eq)]
enum Keep {
    Echo,
    Heartbeat,
}

/// What keeps the link of the current connection awake.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
enum Mode {
    /// ECHO requests (until `MAX_UNANSWERED` go without an answer)
    Echo,
    /// The client does not answer ECHO: Heartbeat PDUs
    Heartbeat,
    /// The client takes neither: nothing
    Stopped,
}

/// The state of the timer task.
struct KeepAwakeLoop {
    shared: KeepAwake,
    last_output: Instant,
    unanswered: u32,
    /// The connection the counters belong to
    connection: u64,
}

impl KeepAwakeLoop {
    /// `now`: the task starts (no write seen yet).
    fn new(shared: KeepAwake, now: Instant) -> Self {
        let connection = shared.connection.load(Ordering::Relaxed);
        Self {
            shared,
            last_output: now,
            unanswered: 0,
            connection,
        }
    }

    /// Something went out to the client at `at`.
    fn on_output(&mut self, at: Instant) {
        self.last_output = self.last_output.max(at);
    }

    /// When to check again; None while there is nothing to keep awake.
    fn due(&self, now: Instant) -> Option<Instant> {
        (self.shared.mode() != Mode::Stopped && self.shared.input_recent(now))
            .then_some(self.last_output + INTERVAL)
    }

    /// Send a request if nothing went out for `INTERVAL`.
    fn tick(&mut self, now: Instant) {
        let connection = self.shared.connection.load(Ordering::Relaxed);
        if connection != self.connection {
            self.connection = connection;
            self.unanswered = 0;
        }
        if self.shared.mode() == Mode::Stopped
            || !self.shared.input_recent(now)
            || now < self.last_output + INTERVAL
        {
            return;
        }
        self.last_output = now;
        let echo = self.shared.echo.get().cloned();
        let answers = echo.as_ref().map_or(0, |e| e.take_measurements().len());
        match self.choose(answers, self.shared.echo_open()) {
            Some(Keep::Echo) => {
                if let Some(Err(e)) = echo.map(|echo| echo.send_request(vec![0x6b])) {
                    tracing::debug!("Keep-awake ECHO request not sent: {e}");
                }
            }
            Some(Keep::Heartbeat) => {
                self.shared.send_heartbeat();
            }
            None => {}
        }
    }

    /// What goes out at this tick, given the ECHO answers since the last one
    /// and whether the client has the ECHO channel open.
    fn choose(&mut self, answers: usize, echo_open: bool) -> Option<Keep> {
        match self.shared.mode() {
            Mode::Heartbeat => return Some(Keep::Heartbeat),
            Mode::Stopped => return None,
            Mode::Echo => {}
        }
        if !echo_open {
            // Not open yet, or declined: no ECHO request, and no count against it.
            if !self.shared.client_heartbeat.load(Ordering::Relaxed) {
                return None;
            }
            if self.shared.first_heartbeat_without_echo() {
                tracing::info!(
                    "ECHO channel not open: Wi-Fi keep-awake sends Heartbeat PDUs until it opens"
                );
            }
            return Some(Keep::Heartbeat);
        }
        if !self.count_answer(answers) {
            return Some(Keep::Echo);
        }
        if self.shared.client_heartbeat.load(Ordering::Relaxed) {
            tracing::info!(
                requests = MAX_UNANSWERED,
                "Client does not answer ECHO requests: Wi-Fi keep-awake sends Heartbeat PDUs"
            );
            self.shared.set_mode(Mode::Heartbeat);
            return Some(Keep::Heartbeat);
        }
        tracing::info!(
            requests = MAX_UNANSWERED,
            "Client does not answer ECHO requests: Wi-Fi keep-awake stopped"
        );
        self.shared.set_mode(Mode::Stopped);
        None
    }

    /// Count one request against the answers that came back since the last
    /// one. Returns true when ECHO requests go without an answer.
    fn count_answer(&mut self, answers: usize) -> bool {
        if answers > 0 {
            self.unanswered = 0;
        }
        self.unanswered += 1;
        self.unanswered > MAX_UNANSWERED
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
        (KeepAwakeLoop::new(shared, now), now)
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
        // An older write does not move it back.
        l.on_output(now);
        assert_eq!(l.due(now), Some(now + Duration::from_millis(30) + INTERVAL));
    }

    #[test]
    fn nothing_due_long_after_the_last_input() {
        let (l, now) = looped(true);
        assert!(l.due(now + AFTER_INPUT - Duration::from_secs(1)).is_some());
        assert_eq!(l.due(now + AFTER_INPUT + Duration::from_secs(1)), None);
    }

    #[test]
    fn nothing_due_after_the_connection_ended() {
        let (l, now) = looped(true);
        l.shared.on_disconnect();
        assert_eq!(l.due(now), None);
        l.shared.note_input();
        assert!(
            l.due(now).is_some(),
            "the next client's input starts it again"
        );
    }

    #[test]
    fn stops_when_the_client_does_not_answer() {
        let (mut l, now) = looped(true);
        for _ in 0..MAX_UNANSWERED {
            assert!(!l.count_answer(0));
        }
        assert!(l.count_answer(0));
        assert_eq!(l.choose(0, true), None);
        assert_eq!(l.due(now), None);
    }

    #[test]
    fn heartbeats_after_unanswered_echo_if_the_client_takes_them() {
        let (mut l, now) = looped(true);
        l.shared.on_connect(true);
        for _ in 0..MAX_UNANSWERED {
            assert_eq!(l.choose(0, true), Some(Keep::Echo));
        }
        for _ in 0..10 {
            assert_eq!(l.choose(0, true), Some(Keep::Heartbeat));
        }
        assert!(l.due(now).is_some());
    }

    #[test]
    fn stops_without_echo_and_heartbeat() {
        let (mut l, now) = looped(true);
        for _ in 0..MAX_UNANSWERED {
            assert_eq!(l.choose(0, true), Some(Keep::Echo));
        }
        assert_eq!(l.choose(0, true), None);
        assert_eq!(l.due(now), None);
    }

    #[test]
    fn a_new_connection_tries_echo_again() {
        let (mut l, now) = looped(true);
        l.shared.on_connect(true);
        for _ in 0..MAX_UNANSWERED {
            assert_eq!(l.choose(0, true), Some(Keep::Echo));
        }
        assert_eq!(l.choose(0, true), Some(Keep::Heartbeat));
        assert_eq!(l.choose(0, true), Some(Keep::Heartbeat));
        // A new connection: ECHO again, with a fresh count (`tick` resets it).
        l.shared.on_connect(true);
        l.tick(now);
        assert_eq!(l.unanswered, 0);
        assert_eq!(l.choose(0, true), Some(Keep::Echo));
    }

    #[test]
    fn stopped_holds_for_the_connection() {
        let (mut l, now) = looped(true);
        for _ in 0..MAX_UNANSWERED {
            assert_eq!(l.choose(0, true), Some(Keep::Echo));
        }
        assert_eq!(l.choose(0, true), None);
        assert_eq!(l.due(now), None);
        l.shared.on_connect(false);
        assert!(l.due(now).is_some());
    }

    #[test]
    fn heartbeat_goes_to_the_handle_once_it_is_set() {
        let shared = KeepAwake::new();
        assert!(!shared.send_heartbeat(), "no handle: nothing to ask");
        shared.set_heartbeat(HeartbeatHandle::default());
        // A request with no client loop is dropped by the server.
        assert!(shared.send_heartbeat());
        // Clones share the handle (the task uses a clone).
        assert!(shared.clone().send_heartbeat());
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

    #[test]
    fn no_echo_before_the_channel_is_open() {
        let (mut l, _) = looped(true);
        // No heartbeats either: nothing goes out, and nothing counts as unanswered.
        for _ in 0..3 * MAX_UNANSWERED {
            assert_eq!(l.choose(0, false), None);
        }
        assert_eq!(l.shared.mode(), Mode::Echo);
        // The channel opens: ECHO gets all its tries.
        for _ in 0..MAX_UNANSWERED {
            assert_eq!(l.choose(0, true), Some(Keep::Echo));
        }
        assert_eq!(l.choose(0, true), None);
    }

    #[test]
    fn heartbeats_until_the_echo_channel_opens() {
        let (mut l, _) = looped(true);
        l.shared.on_connect(true);
        for _ in 0..3 * MAX_UNANSWERED {
            assert_eq!(l.choose(0, false), Some(Keep::Heartbeat));
        }
        assert_eq!(
            l.shared.mode(),
            Mode::Echo,
            "a declined channel is not a choice"
        );
        assert_eq!(l.choose(0, true), Some(Keep::Echo));
        assert_eq!(l.choose(1, true), Some(Keep::Echo));
    }

    #[test]
    fn no_echo_handle_means_no_open_channel() {
        assert!(!KeepAwake::new().echo_open());
    }

    #[test]
    fn heartbeats_without_echo_are_noted_once_per_connection() {
        let (mut l, _) = looped(true);
        l.shared.on_connect(true);
        assert!(!l.shared.heartbeat_without_echo.load(Ordering::Relaxed));
        assert_eq!(l.choose(0, false), Some(Keep::Heartbeat));
        assert!(
            !l.shared.first_heartbeat_without_echo(),
            "noted at the first one"
        );
        // A new connection is noted again.
        l.shared.on_connect(true);
        assert!(l.shared.first_heartbeat_without_echo());
    }
}
