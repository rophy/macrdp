use core::time::Duration;
use std::collections::{BTreeMap, VecDeque};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::Instant;

use anyhow::{Context as _, Result, bail};
use ironrdp_core::impl_as_any;
use ironrdp_dvc::{DvcMessage, DvcProcessor, DvcServerProcessor};
use ironrdp_echo::server::EchoServer;
use ironrdp_pdu::PduResult;
use tokio::sync::mpsc;

use crate::server::ServerEvent;

#[derive(Debug, Clone)]
pub struct EchoRoundTripMeasurement {
    pub payload: Vec<u8>,
    pub round_trip_time: Duration,
}

#[derive(Debug)]
pub enum EchoServerMessage {
    SendRequest { payload: Vec<u8> },
}

impl core::fmt::Display for EchoServerMessage {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::SendRequest { payload } => write!(f, "SendRequest(size={})", payload.len()),
        }
    }
}

#[derive(Debug, Default)]
struct EchoHandleState {
    pending: BTreeMap<Vec<u8>, VecDeque<Instant>>,
    measurements: VecDeque<EchoRoundTripMeasurement>,
    /// (vendored, divergence (30)) Id of the last bridge made (one per connection).
    last_bridge: u64,
    /// Id of the bridge whose channel the client opened, if any.
    open_bridge: Option<u64>,
}

/// Shared handle for runtime ECHO requests and RTT measurements.
#[derive(Debug, Clone)]
pub struct EchoServerHandle {
    sender: mpsc::UnboundedSender<ServerEvent>,
    state: Arc<Mutex<EchoHandleState>>,
}

impl EchoServerHandle {
    pub(crate) fn new(sender: mpsc::UnboundedSender<ServerEvent>) -> Self {
        Self {
            sender,
            state: Arc::new(Mutex::new(EchoHandleState::default())),
        }
    }

    /// Sends a runtime ECHO request.
    ///
    /// The payload must be at least one byte, as required by MS-RDPEECO section 3.1.5.1.
    pub fn send_request(&self, payload: Vec<u8>) -> Result<()> {
        if payload.is_empty() {
            bail!("echoRequest payload must be at least one byte");
        }

        self.sender
            .send(ServerEvent::Echo(EchoServerMessage::SendRequest { payload }))
            .map_err(|_error| anyhow::anyhow!("send ECHO request event"))
    }

    /// (vendored, divergence (30)) True while the client of the current
    /// connection has the ECHO dynamic channel open. False before the client
    /// answered the DVC Create Request, when it declined the channel, and after
    /// the connection ended. The server drops a request sent while this is false.
    pub fn is_channel_open(&self) -> bool {
        self.lock_state().open_bridge.is_some()
    }

    /// Drains collected RTT measurements.
    pub fn take_measurements(&self) -> Vec<EchoRoundTripMeasurement> {
        let mut state = self.lock_state();
        state.measurements.drain(..).collect()
    }

    pub(crate) fn on_request_sent(&self, payload: &[u8]) {
        let mut state = self.lock_state();
        state
            .pending
            .entry(payload.to_vec())
            .or_default()
            .push_back(Instant::now());
    }

    fn on_response(&self, payload: &[u8]) {
        let mut state = self.lock_state();
        let Some(sent_at_queue) = state.pending.get_mut(payload) else {
            return;
        };

        let Some(sent_at) = sent_at_queue.pop_front() else {
            return;
        };

        if sent_at_queue.is_empty() {
            state.pending.remove(payload);
        }

        state.measurements.push_back(EchoRoundTripMeasurement {
            payload: payload.to_vec(),
            round_trip_time: sent_at.elapsed(),
        });
    }

    fn new_bridge_id(&self) -> u64 {
        let mut state = self.lock_state();
        state.last_bridge += 1;
        state.last_bridge
    }

    fn set_open(&self, bridge: u64, open: bool) {
        let mut state = self.lock_state();
        if open {
            state.open_bridge = Some(bridge);
        } else if state.open_bridge == Some(bridge) {
            // A newer connection may have opened its channel already.
            state.open_bridge = None;
        }
    }

    fn lock_state(&self) -> MutexGuard<'_, EchoHandleState> {
        match self.state.lock() {
            Ok(state) => state,
            Err(poisoned) => poisoned.into_inner(),
        }
    }
}

/// DVC bridge for ECHO that tracks RTT on responses.
pub struct EchoDvcBridge {
    inner: EchoServer,
    handle: EchoServerHandle,
    /// Tells this connection's channel state apart from that of another one.
    id: u64,
}

impl EchoDvcBridge {
    pub fn new(handle: EchoServerHandle) -> Self {
        let id = handle.new_bridge_id();
        Self {
            inner: EchoServer::new(),
            handle,
            id,
        }
    }

    pub fn handle(&self) -> &EchoServerHandle {
        &self.handle
    }
}

impl_as_any!(EchoDvcBridge);

impl DvcProcessor for EchoDvcBridge {
    fn channel_name(&self) -> &str {
        self.inner.channel_name()
    }

    fn start(&mut self, channel_id: u32) -> PduResult<Vec<DvcMessage>> {
        self.handle.set_open(self.id, true);
        self.inner.start(channel_id)
    }

    fn process(&mut self, channel_id: u32, payload: &[u8]) -> PduResult<Vec<DvcMessage>> {
        let messages = self.inner.process(channel_id, payload)?;
        self.handle.on_response(payload);
        Ok(messages)
    }

    fn close(&mut self, channel_id: u32) {
        self.handle.set_open(self.id, false);
        self.inner.close(channel_id)
    }
}

impl Drop for EchoDvcBridge {
    // The DVC server calls `close` before it drops an opened channel. This is
    // a guard for the other paths (a declined or pending channel).
    fn drop(&mut self) {
        self.handle.set_open(self.id, false);
    }
}

impl DvcServerProcessor for EchoDvcBridge {}

pub(crate) fn build_echo_request(payload: Vec<u8>) -> Result<DvcMessage> {
    EchoServer::request_message(payload).context("build ECHO request message")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn handle() -> EchoServerHandle {
        let (sender, _receiver) = mpsc::unbounded_channel();
        EchoServerHandle::new(sender)
    }

    #[test]
    fn channel_is_open_from_start_until_the_bridge_goes() {
        let handle = handle();
        let mut bridge = EchoDvcBridge::new(handle.clone());
        assert!(
            !handle.is_channel_open(),
            "the client has not answered the Create Request"
        );
        bridge.start(7).unwrap();
        assert!(handle.is_channel_open());
        drop(bridge);
        assert!(!handle.is_channel_open(), "channel closed or connection ended");
    }

    #[test]
    fn declined_channel_is_not_open() {
        let handle = handle();
        // The client declined: the DVC server never calls `start`.
        let _bridge = EchoDvcBridge::new(handle.clone());
        assert!(!handle.is_channel_open());
    }

    #[test]
    fn old_connection_does_not_close_the_new_channel() {
        let handle = handle();
        let mut old = EchoDvcBridge::new(handle.clone());
        old.start(7).unwrap();
        let mut new = EchoDvcBridge::new(handle.clone());
        new.start(9).unwrap();
        drop(old);
        assert!(handle.is_channel_open(), "the new connection's channel stays open");
        new.close(9);
        assert!(!handle.is_channel_open());
    }
}
