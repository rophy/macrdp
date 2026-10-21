//! Disconnect a client that sent no input for `--idle-timeout` seconds.
//!
//! The client gets a Set Error Info PDU with ERRINFO_IDLE_TIMEOUT
//! (MS-RDPBCGR 2.2.5.1.1), so mstsc tells the user why the session ended.
//!
//! Ported from macrdp-v1 (crates/macrdp-server/src/idle.rs, commits 60f630b,
//! 348bb63), relicensed to MIT OR Apache-2.0 by the author.

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use ironrdp_pdu::rdp::server_error_info::{ErrorInfo, ProtocolIndependentCode};
use ironrdp_server::ServerEvent;
use tokio::sync::mpsc::UnboundedSender;

/// Shared between the connection handler, the input handler and the timer task.
#[derive(Clone)]
pub struct IdleTimeout {
    timeout: Duration,
    /// Time of the last input or connect, in ms since `epoch`
    last_input_ms: Arc<AtomicU64>,
    /// A client is connected and was not disconnected for idleness yet
    armed: Arc<AtomicBool>,
    epoch: Instant,
}

impl IdleTimeout {
    /// `secs` = 0: off (None).
    pub fn new(secs: u64) -> Option<Self> {
        (secs > 0).then(|| Self {
            timeout: Duration::from_secs(secs),
            last_input_ms: Arc::new(AtomicU64::new(0)),
            armed: Arc::new(AtomicBool::new(false)),
            epoch: Instant::now(),
        })
    }

    fn now_ms(&self, now: Instant) -> u64 {
        now.saturating_duration_since(self.epoch).as_millis() as u64
    }

    /// A new client connected: its idle time starts now.
    pub fn on_connect(&self) {
        self.on_connect_at(Instant::now());
    }

    fn on_connect_at(&self, now: Instant) {
        self.last_input_ms
            .store(self.now_ms(now), Ordering::Relaxed);
        self.armed.store(true, Ordering::Relaxed);
    }

    /// The client disconnected: no timeout until the next client connects.
    /// Without this, the old deadline would fire with no client connected, or
    /// during the handshake of the next client, and disconnect that client.
    pub fn on_disconnect(&self) {
        self.armed.store(false, Ordering::Relaxed);
    }

    /// The client sent input.
    pub fn note_input(&self) {
        self.note_input_at(Instant::now());
    }

    fn note_input_at(&self, now: Instant) {
        self.last_input_ms
            .store(self.now_ms(now), Ordering::Relaxed);
    }

    /// When to check again (the time the client becomes idle if no input comes).
    fn deadline(&self) -> Instant {
        self.epoch
            + Duration::from_millis(self.last_input_ms.load(Ordering::Relaxed))
            + self.timeout
    }

    /// True once per connection, when the client is idle for the timeout.
    pub(crate) fn expired_at(&self, now: Instant) -> bool {
        self.armed.load(Ordering::Relaxed)
            && now >= self.deadline()
            && self.armed.swap(false, Ordering::Relaxed)
    }

    /// Timer task: disconnect the client when it is idle for the timeout.
    pub async fn run(self, events: UnboundedSender<ServerEvent>) {
        loop {
            let now = Instant::now();
            if self.expired_at(now) {
                tracing::info!(
                    timeout_secs = self.timeout.as_secs(),
                    "No input for the idle timeout: disconnecting the client"
                );
                let reason =
                    ErrorInfo::ProtocolIndependentCode(ProtocolIndependentCode::IdleTimeout);
                if events.send(ServerEvent::Disconnect(reason)).is_err() {
                    return; // server stopped
                }
            }
            // Wake at the deadline, but at least each second (a new connect or
            // input moves the deadline; checking late by up to 1 s is fine).
            let wait = self
                .deadline()
                .saturating_duration_since(now)
                .clamp(Duration::from_millis(100), Duration::from_secs(1));
            tokio::time::sleep(wait).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_is_off() {
        assert!(IdleTimeout::new(0).is_none());
    }

    #[test]
    fn nothing_before_a_client_connects() {
        let idle = IdleTimeout::new(10).unwrap();
        assert!(!idle.expired_at(Instant::now() + Duration::from_secs(3600)));
    }

    #[test]
    fn expires_once_after_the_timeout_without_input() {
        let idle = IdleTimeout::new(10).unwrap();
        let t0 = Instant::now();
        idle.on_connect_at(t0);
        assert!(!idle.expired_at(t0 + Duration::from_secs(9)));
        assert!(idle.expired_at(t0 + Duration::from_secs(10)));
        // Not again for the same connection.
        assert!(!idle.expired_at(t0 + Duration::from_secs(30)));
        // A new connection starts a new timeout.
        idle.on_connect_at(t0 + Duration::from_secs(40));
        assert!(!idle.expired_at(t0 + Duration::from_secs(45)));
        assert!(idle.expired_at(t0 + Duration::from_secs(50)));
    }

    #[test]
    fn nothing_after_the_client_disconnects() {
        let idle = IdleTimeout::new(10).unwrap();
        let t0 = Instant::now();
        idle.on_connect_at(t0);
        idle.on_disconnect();
        // The old deadline passes with no client, and during the next handshake.
        assert!(!idle.expired_at(t0 + Duration::from_secs(10)));
        assert!(!idle.expired_at(t0 + Duration::from_secs(60)));
        // The next client gets its own full timeout.
        idle.on_connect_at(t0 + Duration::from_secs(60));
        assert!(!idle.expired_at(t0 + Duration::from_secs(69)));
        assert!(idle.expired_at(t0 + Duration::from_secs(70)));
    }

    #[tokio::test]
    async fn timer_sends_nothing_after_a_disconnect() {
        let idle = IdleTimeout::new(1).unwrap();
        idle.on_connect();
        idle.on_disconnect();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let task = tokio::spawn(idle.clone().run(tx));
        let ev = tokio::time::timeout(Duration::from_millis(2500), rx.recv()).await;
        task.abort();
        assert!(ev.is_err(), "unexpected event: {ev:?}");
    }

    #[test]
    fn input_restarts_the_timeout() {
        let idle = IdleTimeout::new(10).unwrap();
        let t0 = Instant::now();
        idle.on_connect_at(t0);
        idle.note_input_at(t0 + Duration::from_secs(8));
        assert!(!idle.expired_at(t0 + Duration::from_secs(12)));
        assert!(idle.expired_at(t0 + Duration::from_secs(18)));
    }

    #[tokio::test]
    async fn timer_sends_an_idle_timeout_disconnect() {
        let idle = IdleTimeout::new(1).unwrap();
        idle.on_connect();
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
        let task = tokio::spawn(idle.run(tx));
        let ev = tokio::time::timeout(Duration::from_secs(3), rx.recv()).await;
        task.abort();
        assert!(matches!(
            ev,
            Ok(Some(ServerEvent::Disconnect(
                ErrorInfo::ProtocolIndependentCode(ProtocolIndependentCode::IdleTimeout)
            )))
        ));
    }
}
