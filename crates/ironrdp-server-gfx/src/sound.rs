pub use ironrdp_rdpsnd::server::{RdpsndServerHandler, RdpsndServerMessage};

use crate::ServerEventSender;

pub trait SoundServerFactory: ServerEventSender {
    fn build_backend(&self) -> Box<dyn RdpsndServerHandler>;

    /// Called after logon when the client asked for no audio playback
    /// (NO_AUDIO_PLAYBACK or REMOTE_CONSOLE_AUDIO). The rdpsnd channel is
    /// not started for this connection; the backend can stop its audio source.
    fn playback_disabled(&self) {}
}
