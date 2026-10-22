//! Per-session features that follow the client's input and the connection
//! edges: the Wi-Fi keep-awake (`keep_awake`) and the idle timeout (`idle`).

use std::net::SocketAddr;
use std::time::Duration;

use ironrdp_server::{ClientSession, ConnectionHandler, PostConnectionAction};

use crate::idle::IdleTimeout;
use crate::keep_awake::KeepAwake;

/// What the input handler tells about each client input.
#[derive(Clone, Default)]
pub struct SessionInput {
    pub keep_awake: Option<KeepAwake>,
    pub idle: Option<IdleTimeout>,
}

impl SessionInput {
    /// The client sent keyboard or mouse input.
    pub fn note_input(&self) {
        if let Some(k) = &self.keep_awake {
            k.note_input();
        }
        if let Some(idle) = &self.idle {
            idle.note_input();
        }
    }
}

/// Wraps the connection handler: tells the session features when a client
/// connects and when its connection ends. Every hook is passed on unchanged.
pub struct SessionHandler {
    inner: Option<Box<dyn ConnectionHandler>>,
    session: SessionInput,
}

impl SessionHandler {
    pub fn wrap(
        inner: Option<Box<dyn ConnectionHandler>>,
        session: SessionInput,
    ) -> Box<dyn ConnectionHandler> {
        Box::new(Self { inner, session })
    }
}

impl ConnectionHandler for SessionHandler {
    fn on_accept(&mut self, peer: SocketAddr) -> bool {
        self.inner.as_mut().is_none_or(|h| h.on_accept(peer))
    }

    fn on_disconnected(
        &mut self,
        peer: SocketAddr,
        duration: Duration,
        error: Option<&anyhow::Error>,
    ) -> PostConnectionAction {
        if let Some(k) = &self.session.keep_awake {
            k.on_disconnect();
        }
        if let Some(idle) = &self.session.idle {
            idle.on_disconnect();
        }
        self.inner
            .as_mut()
            .map_or(PostConnectionAction::Continue, |h| {
                h.on_disconnected(peer, duration, error)
            })
    }

    fn on_authenticated(&mut self, success: bool, reason: Option<&str>) {
        if let Some(h) = self.inner.as_mut() {
            h.on_authenticated(success, reason);
        }
    }

    fn on_client_fingerprint(
        &mut self,
        client_name: &str,
        rdp_version: u32,
        client_build: u32,
        platform: &str,
    ) {
        if let Some(h) = self.inner.as_mut() {
            h.on_client_fingerprint(client_name, rdp_version, client_build, platform);
        }
    }

    fn on_client_session(&mut self, session: &ClientSession) {
        crate::cursor::cache::set_client_cache_size(session.pointer_cache_size);
        if let Some(k) = &self.session.keep_awake {
            k.on_connect(session.supports_heartbeat);
        }
        if let Some(idle) = &self.session.idle {
            idle.on_connect();
        }
        if let Some(h) = self.inner.as_mut() {
            h.on_client_session(session);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct Rejecting;
    impl ConnectionHandler for Rejecting {
        fn on_accept(&mut self, _peer: SocketAddr) -> bool {
            false
        }
        fn on_disconnected(
            &mut self,
            _peer: SocketAddr,
            _duration: Duration,
            _error: Option<&anyhow::Error>,
        ) -> PostConnectionAction {
            PostConnectionAction::Stop
        }
    }

    #[test]
    fn hooks_pass_through() {
        let peer: SocketAddr = "127.0.0.1:1".parse().unwrap();
        let mut h = SessionHandler::wrap(
            Some(Box::new(Rejecting)),
            SessionInput {
                keep_awake: Some(KeepAwake::new()),
                idle: None,
            },
        );
        assert!(!h.on_accept(peer), "the inner verdict is kept");
        assert_eq!(
            h.on_disconnected(peer, Duration::from_secs(1), None),
            PostConnectionAction::Stop
        );
        let mut open = SessionHandler::wrap(None, SessionInput::default());
        assert!(open.on_accept(peer));
        open.on_client_session(&ClientSession::default());
        assert_eq!(
            open.on_disconnected(peer, Duration::from_secs(1), None),
            PostConnectionAction::Continue
        );
    }

    /// The idle timeout runs from the client session start to the
    /// disconnect, through the handler.
    #[test]
    fn the_idle_timeout_follows_the_connection() {
        let peer: SocketAddr = "127.0.0.1:1".parse().unwrap();
        let idle = IdleTimeout::new(10).unwrap();
        let mut h = SessionHandler::wrap(
            None,
            SessionInput {
                keep_awake: None,
                idle: Some(idle.clone()),
            },
        );
        let later = std::time::Instant::now() + Duration::from_secs(20);
        assert!(!idle.expired_at(later), "no client yet");
        h.on_client_session(&ClientSession::default());
        h.on_disconnected(peer, Duration::from_secs(1), None);
        assert!(!idle.expired_at(later), "no client: no idle disconnect");
        h.on_client_session(&ClientSession::default());
        assert!(idle.expired_at(later), "the next client is timed");
    }
}
