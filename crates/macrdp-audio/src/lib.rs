pub mod converter;

use std::collections::VecDeque;
use std::fmt;

use ironrdp_rdpsnd::pdu::{AudioFormat, WaveFormat};
use ironrdp_rdpsnd::server::{
    NegotiatedFormat, RdpsndError, RdpsndServerHandler, RdpsndServerMessage,
};
use ironrdp_server::{
    audio_playback_wanted, ConnectionHandler, ConnectionInfo, ServerEvent, ServerEventSender,
    SoundServerFactory,
};
use macrdp_capture::AudioFrame;
use tokio::sync::mpsc;

/// Audio processing loop: receives AudioFrames from SCK, converts to S16LE,
/// and sends RdpsndServerMessage::Wave events.
async fn audio_loop(
    mut audio_rx: mpsc::Receiver<AudioFrame>,
    event_sender: tokio::sync::mpsc::UnboundedSender<ServerEvent>,
    frame_size_interleaved: usize,
    channels: u16,
    sample_rate: u32,
) {
    let mut ring_buffer: VecDeque<f32> = VecDeque::with_capacity(frame_size_interleaved * 4);
    let mut base_timestamp_ms: Option<u64> = None;
    let mut samples_sent: u64 = 0;

    tracing::info!(
        frame_size_interleaved,
        "Audio loop started ({}Hz, {}ch)",
        sample_rate,
        channels
    );

    while let Some(frame) = audio_rx.recv().await {
        if base_timestamp_ms.is_none() {
            base_timestamp_ms = Some(frame.timestamp_ms);
        }
        ring_buffer.extend(frame.data.iter());

        while ring_buffer.len() >= frame_size_interleaved {
            let chunk: Vec<f32> = ring_buffer.drain(..frame_size_interleaved).collect();
            let wave_data = converter::AudioConverter::float32_to_s16le(&chunk);

            let samples_per_ch = frame_size_interleaved / channels as usize;
            let offset_ms = (samples_sent * 1000) / sample_rate as u64;
            let ts = base_timestamp_ms.unwrap_or(0) + offset_ms;
            samples_sent += samples_per_ch as u64;

            // SCK delivers buffers even when nothing plays. Sending digital
            // silence as PCM costs ~1.5 Mbps, so skip it; the timestamp of the
            // next sent chunk still follows the real timeline.
            if is_silent(&wave_data) {
                continue;
            }

            let _ = event_sender.send(ServerEvent::Rdpsnd(RdpsndServerMessage::Wave(
                wave_data, ts as u32,
            )));
        }
    }
    tracing::info!("Audio loop ended");
}

/// True when every S16LE sample in the chunk is zero.
fn is_silent(wave_data: &[u8]) -> bool {
    wave_data.iter().all(|&b| b == 0)
}

pub struct MacAudioHandler {
    /// The one format we offer: PCM S16LE at the configured rate and channel count.
    format: AudioFormat,
    started: bool,
}

impl fmt::Debug for MacAudioHandler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MacAudioHandler")
            .field("started", &self.started)
            .finish_non_exhaustive()
    }
}

impl MacAudioHandler {
    pub fn new(sample_rate: u32, channels: u16) -> Self {
        let format = AudioFormat {
            format: WaveFormat::PCM,
            n_channels: channels,
            n_samples_per_sec: sample_rate,
            n_avg_bytes_per_sec: sample_rate * (channels as u32) * 2,
            n_block_align: channels * 2,
            bits_per_sample: 16,
            data: None,
        };

        Self {
            format,
            started: false,
        }
    }
}

impl RdpsndServerHandler for MacAudioHandler {
    fn get_formats(&self) -> &[AudioFormat] {
        std::slice::from_ref(&self.format)
    }

    fn choose_format<'a>(
        &mut self,
        common: &'a [NegotiatedFormat],
    ) -> Option<&'a NegotiatedFormat> {
        // We offer one format; take it if the client has it.
        common.iter().find(|f| *f.format() == self.format)
    }

    fn start(&mut self, _format: &NegotiatedFormat) -> Result<(), Box<dyn RdpsndError>> {
        self.started = true;
        tracing::info!(
            sample_rate = self.format.n_samples_per_sec,
            channels = self.format.n_channels,
            "Audio started"
        );
        Ok(())
    }

    fn stop(&mut self) {
        tracing::info!("Audio stopped");
        self.started = false;
    }
}

/// Shared audio sender that can be swapped on each connection.
/// Display reads the current sender; AudioFactory replaces it with a fresh one per connection.
pub type SharedAudioTx = std::sync::Arc<std::sync::Mutex<Option<mpsc::Sender<AudioFrame>>>>;

/// Create a shared audio sender slot for use by both MacDisplay and MacAudioFactory.
pub fn new_shared_audio_tx() -> SharedAudioTx {
    std::sync::Arc::new(std::sync::Mutex::new(None))
}

/// Factory that creates MacAudioHandler instances.
/// Creates a fresh audio channel for each connection so the server survives reconnects.
pub struct MacAudioFactory {
    shared_tx: SharedAudioTx,
    event_sender: Option<tokio::sync::mpsc::UnboundedSender<ServerEvent>>,
    sample_rate: u32,
    channels: u16,
}

impl MacAudioFactory {
    pub fn new(shared_tx: SharedAudioTx, sample_rate: u32, channels: u16) -> Self {
        Self {
            shared_tx,
            event_sender: None,
            sample_rate,
            channels,
        }
    }
}

impl SoundServerFactory for MacAudioFactory {
    fn build_backend(&self) -> Box<dyn RdpsndServerHandler> {
        let sender = self.event_sender.clone().expect("event sender not set");

        let channels = self.channels;
        let sample_rate = self.sample_rate;
        // 20ms frame: sample_rate/50 samples per channel * channels
        let frame_size_interleaved = (sample_rate as usize / 50) * channels as usize;

        // Create a fresh channel for this connection
        let (tx, rx) = mpsc::channel::<AudioFrame>(32);
        // Store the new sender so the display capturer uses it
        *self.shared_tx.lock().unwrap() = Some(tx);

        tokio::spawn(audio_loop(
            rx,
            sender,
            frame_size_interleaved,
            channels,
            sample_rate,
        ));

        Box::new(MacAudioHandler::new(sample_rate, channels))
    }
}

/// Stops audio capture for a client that does not want audio played on its side.
///
/// The server builds the rdpsnd backend (and so a fresh audio sender) for every
/// connection, but does not start rdpsnd when the client set INFO_NOAUDIOPLAYBACK or
/// INFO_REMOTECONSOLEAUDIO. The server calls `on_connection_info` before the display
/// starts capture, so dropping the sender here makes the display capture without audio,
/// and `audio_loop` ends when its receiver closes.
pub struct AudioPlaybackGate {
    shared_tx: SharedAudioTx,
}

impl AudioPlaybackGate {
    pub fn new(shared_tx: SharedAudioTx) -> Self {
        Self { shared_tx }
    }
}

impl ConnectionHandler for AudioPlaybackGate {
    fn on_connection_info(&mut self, info: &ConnectionInfo) {
        if !audio_playback_wanted(info.client_info_flags) {
            tracing::info!("Client asked for no audio playback, audio capture off");
            *self.shared_tx.lock().unwrap() = None;
        }
    }
}

impl ServerEventSender for MacAudioFactory {
    fn set_sender(&mut self, sender: tokio::sync::mpsc::UnboundedSender<ServerEvent>) {
        self.event_sender = Some(sender);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ironrdp_core::encode_vec;
    use ironrdp_pdu::gcc::KeyboardType;
    use ironrdp_pdu::rdp::client_info::ClientInfoFlags;
    use ironrdp_rdpsnd::pdu::{
        AudioFormatFlags, ClientAudioFormatPdu, ClientAudioOutputPdu, TrainingConfirmPdu, Version,
    };
    use ironrdp_rdpsnd::server::RdpsndServer;
    use ironrdp_svc::SvcProcessor;

    fn pcm(n_channels: u16, n_samples_per_sec: u32) -> AudioFormat {
        AudioFormat {
            format: WaveFormat::PCM,
            n_channels,
            n_samples_per_sec,
            n_avg_bytes_per_sec: n_samples_per_sec * u32::from(n_channels) * 2,
            n_block_align: n_channels * 2,
            bits_per_sample: 16,
            data: None,
        }
    }

    /// Run the rdpsnd negotiation against our handler with the given client formats,
    /// and return the server (a wave can be sent only if a format was chosen).
    fn negotiate(client_formats: Vec<AudioFormat>) -> RdpsndServer {
        let mut server = RdpsndServer::new(Box::new(MacAudioHandler::new(48000, 2)));
        server.start().unwrap();
        let formats = ClientAudioOutputPdu::AudioFormat(ClientAudioFormatPdu {
            version: Version::V5,
            flags: AudioFormatFlags::ALIVE,
            formats: client_formats,
            volume_left: 0xFFFF,
            volume_right: 0xFFFF,
            pitch: 0,
            dgram_port: 0,
        });
        server.process(&encode_vec(&formats).unwrap()).unwrap();
        let confirm = ClientAudioOutputPdu::TrainingConfirm(TrainingConfirmPdu {
            timestamp: 0,
            pack_size: 0,
        });
        server.process(&encode_vec(&confirm).unwrap()).unwrap();
        server
    }

    #[test]
    fn silent_chunk_detected() {
        assert!(is_silent(&[0u8; 8]));
        assert!(!is_silent(&[0, 0, 1, 0]));
    }

    #[test]
    fn handler_offers_pcm_s16() {
        let handler = MacAudioHandler::new(48000, 2);
        assert_eq!(handler.get_formats(), &[pcm(2, 48000)]);
    }

    #[test]
    fn format_chosen_when_client_has_it() {
        let mut server = negotiate(vec![pcm(1, 44100), pcm(2, 48000)]);
        assert!(server.wave(vec![0; 16], 0).is_ok());
    }

    #[test]
    fn no_format_when_client_lacks_it() {
        let mut server = negotiate(vec![pcm(1, 44100)]);
        assert!(server.wave(vec![0; 16], 0).is_err());
    }

    fn info_with(flags: ClientInfoFlags) -> ConnectionInfo {
        let mut info = ConnectionInfo::new(0x409, KeyboardType::IBM_ENHANCED, String::new());
        info.client_info_flags = flags;
        info
    }

    #[test]
    fn gate_drops_audio_sender_when_client_wants_no_playback() {
        for flags in [
            ClientInfoFlags::NO_AUDIO_PLAYBACK,
            ClientInfoFlags::REMOTE_CONSOLE_AUDIO,
        ] {
            let shared = new_shared_audio_tx();
            let (tx, _rx) = mpsc::channel(1);
            *shared.lock().unwrap() = Some(tx);
            AudioPlaybackGate::new(shared.clone()).on_connection_info(&info_with(flags));
            assert!(shared.lock().unwrap().is_none(), "{flags:?}");
        }
    }

    #[test]
    fn gate_keeps_audio_sender_when_client_wants_playback() {
        let shared = new_shared_audio_tx();
        let (tx, _rx) = mpsc::channel(1);
        *shared.lock().unwrap() = Some(tx);
        AudioPlaybackGate::new(shared.clone()).on_connection_info(&info_with(
            ClientInfoFlags::MOUSE | ClientInfoFlags::UNICODE,
        ));
        assert!(shared.lock().unwrap().is_some());
    }
}
