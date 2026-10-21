//! (vendored, divergence (30)) On-demand Server Heartbeat PDUs (MS-RDPBCGR
//! 2.2.16.1) and the time of the last write to the client.
//!
//! Both exist for an application that wants to keep the link busy (macrdp's
//! Wi-Fi keep-awake): it needs to know when the server last sent something,
//! and a small PDU to send that the client does not answer.
//!
//! `HeartbeatHandle` is ported from macrdp-v1's IronRDP fork (rophy/IronRDP
//! cd5ac8f4, 639fce30), relicensed by the author. `HeartbeatConfig` follows
//! upstream IronRDP (8f57691a, #1842).

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use ironrdp_core::{Encode as _, WriteCursor, encode_vec};
use ironrdp_pdu::mcs::SendDataIndication;
use ironrdp_pdu::rdp::headers::{BasicSecurityHeader, BasicSecurityHeaderFlags};
use ironrdp_pdu::x224::X224;
use tokio::sync::Notify;

/// Values of a Server Heartbeat PDU (MS-RDPBCGR 2.2.16.1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HeartbeatConfig {
    /// Seconds between heartbeats.
    pub period_secs: u8,
    /// Missed heartbeats that SHOULD trigger a client-side warning.
    pub warning_count: u8,
    /// Missed heartbeats after the warning that SHOULD trigger a client-side
    /// reconnect attempt.
    pub reconnect_count: u8,
}

/// Cloneable handle that sends one Server Heartbeat PDU on demand, while
/// [`RdpServer::run`](crate::RdpServer::run) owns the server. Get it from
/// [`RdpServer::heartbeat_handle`](crate::RdpServer::heartbeat_handle).
/// The client sends no reply.
#[derive(Clone, Default)]
pub struct HeartbeatHandle {
    inner: Arc<HeartbeatRequest>,
}

#[derive(Default)]
struct HeartbeatRequest {
    pending: Mutex<Option<HeartbeatConfig>>,
    notify: Notify,
}

impl HeartbeatHandle {
    /// Send one Heartbeat PDU with `config` to the current client, if it
    /// advertised heartbeat support and has a message channel; otherwise
    /// nothing is sent.
    ///
    /// Requests that come before the server took the last one are merged into
    /// one PDU with the newest `config`. A request made while no client is
    /// connected is dropped when the next client loop starts.
    pub fn send(&self, config: HeartbeatConfig) {
        *self.lock() = Some(config);
        self.inner.notify.notify_one();
    }

    /// Takes the pending request, if any.
    pub(crate) fn take(&self) -> Option<HeartbeatConfig> {
        self.lock().take()
    }

    /// Waits for the next request. Cancel safe: a request is taken only when
    /// this returns.
    pub(crate) async fn requested(&self) -> HeartbeatConfig {
        loop {
            if let Some(config) = self.take() {
                return config;
            }
            self.inner.notify.notified().await;
        }
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Option<HeartbeatConfig>> {
        match self.inner.pending.lock() {
            Ok(guard) => guard,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

impl core::fmt::Debug for HeartbeatHandle {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HeartbeatHandle").finish_non_exhaustive()
    }
}

/// Encode a Server Heartbeat PDU for the MCS message channel: a Basic Security
/// Header with SEC_HEARTBEAT, then reserved (0), period, count1, count2
/// (MS-RDPBCGR 2.2.16.1).
pub(crate) fn encode_heartbeat(
    config: &HeartbeatConfig,
    message_channel_id: u16,
    user_channel_id: u16,
) -> anyhow::Result<Vec<u8>> {
    let header = BasicSecurityHeader {
        flags: BasicSecurityHeaderFlags::HEARTBEAT,
    };
    let mut user_data = vec![0u8; 8];
    let mut cursor = WriteCursor::new(&mut user_data);
    header.encode(&mut cursor)?;
    cursor.write_u8(0);
    cursor.write_u8(config.period_secs);
    cursor.write_u8(config.warning_count);
    cursor.write_u8(config.reconnect_count);
    let pdu = SendDataIndication {
        initiator_id: user_channel_id,
        channel_id: message_channel_id,
        user_data: user_data.into(),
    };
    Ok(encode_vec(&X224(pdu))?)
}

/// Time of the last write to the client socket of the current connection.
/// Get it from [`RdpServer::output_activity`](crate::RdpServer::output_activity).
#[derive(Clone, Debug)]
pub struct OutputActivity {
    inner: Arc<OutputActivityInner>,
}

#[derive(Debug)]
struct OutputActivityInner {
    epoch: Instant,
    /// Milliseconds since `epoch` plus one; 0 = no write yet.
    last_ms: AtomicU64,
}

impl Default for OutputActivity {
    fn default() -> Self {
        Self {
            inner: Arc::new(OutputActivityInner {
                epoch: Instant::now(),
                last_ms: AtomicU64::new(0),
            }),
        }
    }
}

impl OutputActivity {
    pub(crate) fn note_write(&self) {
        let ms = u64::try_from(self.inner.epoch.elapsed().as_millis()).unwrap_or(u64::MAX - 1);
        self.inner.last_ms.store(ms + 1, Ordering::Relaxed);
    }

    /// When the server last wrote to the client (millisecond resolution), if
    /// it wrote at all since the server started.
    pub fn last_write(&self) -> Option<Instant> {
        match self.inner.last_ms.load(Ordering::Relaxed) {
            0 => None,
            ms => Some(self.inner.epoch + Duration::from_millis(ms - 1)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: HeartbeatConfig = HeartbeatConfig {
        period_secs: 255,
        warning_count: 255,
        reconnect_count: 255,
    };

    #[tokio::test]
    async fn a_request_is_taken_once() {
        let handle = HeartbeatHandle::default();
        handle.clone().send(HeartbeatConfig {
            period_secs: 5,
            warning_count: 3,
            reconnect_count: 5,
        });
        handle.send(CONFIG);
        assert_eq!(handle.requested().await, CONFIG, "the newest config wins");
        let next = tokio::time::timeout(Duration::from_millis(20), handle.requested()).await;
        assert!(next.is_err(), "two requests before a take make one PDU");
    }

    #[tokio::test]
    async fn a_taken_request_is_not_returned_again() {
        let handle = HeartbeatHandle::default();
        handle.send(CONFIG);
        assert_eq!(handle.take(), Some(CONFIG));
        let next = tokio::time::timeout(Duration::from_millis(20), handle.requested()).await;
        assert!(next.is_err());
    }

    #[test]
    fn heartbeat_pdu_bytes() {
        let config = HeartbeatConfig {
            period_secs: 5,
            warning_count: 3,
            reconnect_count: 7,
        };
        let bytes = encode_heartbeat(&config, 1004, 1002).unwrap();
        // The PDU is the last 8 bytes of the MCS user data.
        assert_eq!(&bytes[bytes.len() - 8..], &[0x00, 0x40, 0x00, 0x00, 0x00, 5, 3, 7]);
    }

    #[test]
    fn output_activity_records_the_last_write() {
        let activity = OutputActivity::default();
        assert_eq!(activity.last_write(), None);
        let before = Instant::now();
        activity.clone().note_write();
        let at = activity.last_write().expect("a write was noted");
        assert!(at <= Instant::now());
        assert!(before.saturating_duration_since(at) <= Duration::from_millis(1));
    }
}
