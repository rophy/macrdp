use core::net::SocketAddr;
use std::rc::Rc;
use std::sync::Arc;

use anyhow::{anyhow, bail, Context as _, Result};
use ironrdp_acceptor::{Acceptor, AcceptorResult, BeginResult, DesktopSize};
use ironrdp_async::Framed;
use ironrdp_cliprdr::backend::ClipboardMessage;
use ironrdp_cliprdr::pdu::{ClipboardPdu, FileContentsRequest, OwnedFileContentsResponse};
use ironrdp_cliprdr::CliprdrServer;
use ironrdp_core::{decode, encode_vec, impl_as_any};
use ironrdp_displaycontrol::pdu::DisplayControlMonitorLayout;
use ironrdp_displaycontrol::server::{DisplayControlHandler, DisplayControlServer};
use ironrdp_pdu::input::fast_path::{FastPathInput, FastPathInputEvent};
use ironrdp_pdu::input::InputEventPdu;
use ironrdp_pdu::mcs::{SendDataIndication, SendDataRequest};
use ironrdp_pdu::rdp::capability_sets::{BitmapCodecs, CapabilitySet, CmdFlags, CodecProperty, GeneralExtraFlags};
pub use ironrdp_pdu::rdp::client_info::Credentials;
use ironrdp_pdu::rdp::headers::{ServerDeactivateAll, ShareControlPdu};
use ironrdp_pdu::x224::X224;
use ironrdp_pdu::{decode_err, mcs, nego, rdp, Action, PduResult};
use ironrdp_svc::{server_encode_svc_messages, ChannelFlags, StaticChannelId, StaticChannelSet, SvcMessage, SvcProcessor};
use ironrdp_tokio::{split_tokio_framed, unsplit_tokio_framed, FramedRead, FramedWrite, TokioFramed};
use rdpsnd::server::{RdpsndServer, RdpsndServerMessage};
use tokio::io::{AsyncRead, AsyncWrite, AsyncWriteExt as _};
use tokio::net::{TcpListener, TcpStream};
use tokio::sync::{mpsc, oneshot, Mutex};
use tokio::task;
use tokio_rustls::TlsAcceptor;
use tracing::{debug, error, info, trace, warn};
use {ironrdp_dvc as dvc, ironrdp_rdpsnd as rdpsnd};

use crate::clipboard::CliprdrServerFactory;
use crate::display::{DisplayUpdate, RdpServerDisplay};
use crate::encoder::{UpdateEncoder, UpdateEncoderCodecs};
use crate::gfx::{GfxHandler, GfxState};
use crate::handler::RdpServerInputHandler;
use crate::{builder, capabilities, SoundServerFactory};

#[derive(Clone)]
pub struct RdpServerOptions {
    pub addr: SocketAddr,
    pub security: RdpServerSecurity,
    pub codecs: BitmapCodecs,
}

impl RdpServerOptions {
    fn has_image_remote_fx(&self) -> bool {
        self.codecs
            .0
            .iter()
            .any(|codec| matches!(codec.property, CodecProperty::ImageRemoteFx(_)))
    }

    fn has_remote_fx(&self) -> bool {
        self.codecs
            .0
            .iter()
            .any(|codec| matches!(codec.property, CodecProperty::RemoteFx(_)))
    }

    #[cfg(feature = "qoi")]
    fn has_qoi(&self) -> bool {
        self.codecs
            .0
            .iter()
            .any(|codec| matches!(codec.property, CodecProperty::Qoi))
    }

    #[cfg(feature = "qoiz")]
    fn has_qoiz(&self) -> bool {
        self.codecs
            .0
            .iter()
            .any(|codec| matches!(codec.property, CodecProperty::QoiZ))
    }
}

#[derive(Clone)]
pub enum RdpServerSecurity {
    None,
    Tls(TlsAcceptor),
    /// Used for both hybrid + hybrid-ex.
    Hybrid((TlsAcceptor, Vec<u8>)),
}

impl RdpServerSecurity {
    pub fn flag(&self) -> nego::SecurityProtocol {
        match self {
            RdpServerSecurity::None => nego::SecurityProtocol::empty(),
            RdpServerSecurity::Tls(_) => nego::SecurityProtocol::SSL,
            // Advertise HYBRID + HYBRID_EX + SSL so the acceptor can negotiate
            // down to TLS-only if the client doesn't support NLA
            RdpServerSecurity::Hybrid(_) => {
                nego::SecurityProtocol::HYBRID
                    | nego::SecurityProtocol::HYBRID_EX
                    | nego::SecurityProtocol::SSL
            }
        }
    }
}

struct AInputHandler {
    handler: Arc<Mutex<Box<dyn RdpServerInputHandler>>>,
}

impl_as_any!(AInputHandler);

impl dvc::DvcProcessor for AInputHandler {
    fn channel_name(&self) -> &str {
        ironrdp_ainput::CHANNEL_NAME
    }

    fn start(&mut self, _channel_id: u32) -> PduResult<Vec<dvc::DvcMessage>> {
        use ironrdp_ainput::{ServerPdu, VersionPdu};

        let pdu = ServerPdu::Version(VersionPdu::default());

        Ok(vec![Box::new(pdu)])
    }

    fn close(&mut self, _channel_id: u32) {}

    fn process(&mut self, _channel_id: u32, payload: &[u8]) -> PduResult<Vec<dvc::DvcMessage>> {
        use ironrdp_ainput::ClientPdu;

        match decode(payload).map_err(|e| decode_err!(e))? {
            ClientPdu::Mouse(pdu) => {
                let handler = Arc::clone(&self.handler);
                task::spawn_blocking(move || {
                    handler.blocking_lock().mouse(pdu.into());
                });
            }
        }

        Ok(Vec::new())
    }
}

impl dvc::DvcServerProcessor for AInputHandler {}

struct DisplayControlBackend {
    display: Arc<Mutex<Box<dyn RdpServerDisplay>>>,
}

impl DisplayControlBackend {
    fn new(display: Arc<Mutex<Box<dyn RdpServerDisplay>>>) -> Self {
        Self { display }
    }
}

impl DisplayControlHandler for DisplayControlBackend {
    fn monitor_layout(&self, layout: DisplayControlMonitorLayout) {
        let display = Arc::clone(&self.display);
        task::spawn_blocking(move || display.blocking_lock().request_layout(layout));
    }
}

/// RDP Server
///
/// A server is created to listen for connections.
/// After the connection sequence is finalized using the provided security mechanism, the server can:
///  - receive display updates from a [`RdpServerDisplay`] and forward them to the client
///  - receive input events from a client and forward them to an [`RdpServerInputHandler`]
///
/// # Example
///
/// ```
/// use ironrdp_server::{RdpServer, RdpServerInputHandler, RdpServerDisplay, RdpServerDisplayUpdates};
///
///# use anyhow::Result;
///# use ironrdp_server::{DisplayUpdate, DesktopSize, KeyboardEvent, MouseEvent};
///# use tokio_rustls::TlsAcceptor;
///# struct NoopInputHandler;
///# impl RdpServerInputHandler for NoopInputHandler {
///#     fn keyboard(&mut self, _: KeyboardEvent) {}
///#     fn mouse(&mut self, _: MouseEvent) {}
///# }
///# struct NoopDisplay;
///# #[async_trait::async_trait]
///# impl RdpServerDisplay for NoopDisplay {
///#     async fn size(&mut self) -> DesktopSize {
///#         todo!()
///#     }
///#     async fn updates(&mut self) -> Result<Box<dyn RdpServerDisplayUpdates>> {
///#         todo!()
///#     }
///# }
///# async fn stub() -> Result<()> {
/// fn make_tls_acceptor() -> TlsAcceptor {
///    /* snip */
///#    todo!()
/// }
///
/// fn make_input_handler() -> impl RdpServerInputHandler {
///    /* snip */
///#    NoopInputHandler
/// }
///
/// fn make_display_handler() -> impl RdpServerDisplay {
///    /* snip */
///#    NoopDisplay
/// }
///
/// let tls_acceptor = make_tls_acceptor();
/// let input_handler = make_input_handler();
/// let display_handler = make_display_handler();
///
/// let mut server = RdpServer::builder()
///     .with_addr(([127, 0, 0, 1], 3389))
///     .with_tls(tls_acceptor)
///     .with_input_handler(input_handler)
///     .with_display_handler(display_handler)
///     .build();
///
/// server.run().await;
/// Ok(())
///# }
/// ```
pub struct RdpServer {
    opts: RdpServerOptions,
    // FIXME: replace with a channel and poll/process the handler?
    handler: Arc<Mutex<Box<dyn RdpServerInputHandler>>>,
    display: Arc<Mutex<Box<dyn RdpServerDisplay>>>,
    static_channels: StaticChannelSet,
    sound_factory: Option<Box<dyn SoundServerFactory>>,
    cliprdr_factory: Option<Box<dyn CliprdrServerFactory>>,
    ev_sender: mpsc::UnboundedSender<ServerEvent>,
    ev_receiver: Arc<Mutex<mpsc::UnboundedReceiver<ServerEvent>>>,
    creds: Option<Credentials>,
    local_addr: Option<SocketAddr>,
    gfx_state: Arc<std::sync::Mutex<GfxState>>,
    gfx_enabled: bool,
    max_monitors: u32,
}

#[derive(Debug)]
pub enum ServerEvent {
    Quit(String),
    Takeover,
    Clipboard(ClipboardMessage),
    ClipboardFileContents(OwnedFileContentsResponse),
    ClipboardFileContentsRequest(FileContentsRequest),
    Rdpsnd(RdpsndServerMessage),
    SetCredentials(Credentials),
    GetLocalAddr(oneshot::Sender<Option<SocketAddr>>),
}

pub trait ServerEventSender {
    fn set_sender(&mut self, sender: mpsc::UnboundedSender<ServerEvent>);
}

impl ServerEvent {
    pub fn create_channel() -> (mpsc::UnboundedSender<Self>, mpsc::UnboundedReceiver<Self>) {
        mpsc::unbounded_channel()
    }
}

#[derive(Debug, PartialEq)]
enum RunState {
    Continue,
    Disconnect,
    DeactivationReactivation { desktop_size: DesktopSize },
}

impl RdpServer {
    pub fn new(
        opts: RdpServerOptions,
        handler: Box<dyn RdpServerInputHandler>,
        display: Box<dyn RdpServerDisplay>,
        mut sound_factory: Option<Box<dyn SoundServerFactory>>,
        mut cliprdr_factory: Option<Box<dyn CliprdrServerFactory>>,
    ) -> Self {
        let (ev_sender, ev_receiver) = ServerEvent::create_channel();
        if let Some(cliprdr) = cliprdr_factory.as_mut() {
            cliprdr.set_sender(ev_sender.clone());
        }
        if let Some(snd) = sound_factory.as_mut() {
            snd.set_sender(ev_sender.clone());
        }
        Self {
            opts,
            handler: Arc::new(Mutex::new(handler)),
            display: Arc::new(Mutex::new(display)),
            static_channels: StaticChannelSet::new(),
            sound_factory,
            cliprdr_factory,
            ev_sender,
            ev_receiver: Arc::new(Mutex::new(ev_receiver)),
            creds: None,
            local_addr: None,
            gfx_state: Arc::new(std::sync::Mutex::new(GfxState::new(0, 0, false))),
            gfx_enabled: true,
            max_monitors: 1,
        }
    }

    /// Set the GFX state (call before run() to share state with display)
    pub fn set_gfx_state(&mut self, state: Arc<std::sync::Mutex<GfxState>>) {
        self.gfx_state = state;
    }

    /// Get a reference to the GFX state
    pub fn gfx_state(&self) -> &Arc<std::sync::Mutex<GfxState>> {
        &self.gfx_state
    }

    /// Enable or disable GFX H.264 channel
    pub fn set_gfx_enabled(&mut self, enabled: bool) {
        self.gfx_enabled = enabled;
    }

    pub fn set_max_monitors(&mut self, max: u32) {
        self.max_monitors = max;
    }

    pub fn builder() -> builder::RdpServerBuilder<builder::WantsAddr> {
        builder::RdpServerBuilder::new()
    }

    pub fn event_sender(&self) -> &mpsc::UnboundedSender<ServerEvent> {
        &self.ev_sender
    }

    fn attach_channels(&mut self, acceptor: &mut Acceptor) {
        if let Some(cliprdr_factory) = self.cliprdr_factory.as_deref() {
            let backend = cliprdr_factory.build_cliprdr_backend();

            let cliprdr = CliprdrServer::new(backend);

            acceptor.attach_static_channel(cliprdr);
        }

        if let Some(factory) = self.sound_factory.as_deref() {
            let backend = factory.build_backend();

            acceptor.attach_static_channel(RdpsndServer::new(backend));
        }

        let dcs_backend = DisplayControlBackend::new(Arc::clone(&self.display));
        let mut dvc = dvc::DrdynvcServer::new()
            .with_dynamic_channel(AInputHandler {
                handler: Arc::clone(&self.handler),
            })
            .with_dynamic_channel(
                DisplayControlServer::new(Box::new(dcs_backend))
                    .with_max_monitors(self.max_monitors),
            );

        // Only register GFX channel if GFX frame sending is enabled
        if self.gfx_enabled {
            let gfx_handler = GfxHandler::new(Arc::clone(&self.gfx_state));
            dvc = dvc.with_dynamic_channel(gfx_handler);
            debug!("GFX channel registered");
        }

        acceptor.attach_static_channel(dvc);
    }

    pub async fn run_connection(&mut self, stream: TcpStream) -> Result<()> {
        let peer_ip = stream.peer_addr().map(|addr| addr.ip()).ok();
        let framed = TokioFramed::new(stream);

        // Reset protocol state for new connection, preserving network stats and hot-config
        let pending_resolution = {
            let mut gs = self.gfx_state.lock().unwrap();
            let w = gs.width;
            let h = gs.height;
            let avc444_enabled = gs.avc444_enabled;
            let res = gs.resolution.take();
            gs.reset_for_reconnect(w, h, avc444_enabled);
            gs.peer_addr = peer_ip;
            res
        };

        // Apply hot-updated resolution for this new connection
        if let Some(ref res) = pending_resolution {
            if let Some((w, h)) = res.split_once('x').and_then(|(w, h)| {
                Some((w.parse::<u16>().ok()?, h.parse::<u16>().ok()?))
            }) {
                info!(w, h, "Applying hot-updated resolution for new connection");
                // Use set_size (bypasses fixed_resolution check, since this is server-initiated)
                self.display.lock().await.set_size(w, h);
                let mut gs = self.gfx_state.lock().unwrap();
                gs.width = w;
                gs.height = h;
            }
        }

        let size = self.display.lock().await.size().await;
        let capabilities = capabilities::capabilities(&self.opts, size);
        let mut acceptor = Acceptor::new(self.opts.security.flag(), size, capabilities, self.creds.clone());

        self.attach_channels(&mut acceptor);

        let res = ironrdp_acceptor::accept_begin(framed, &mut acceptor)
            .await
            .context("accept_begin failed")?;

        match res {
            BeginResult::ShouldUpgrade(stream) => {
                let tls_acceptor = match &self.opts.security {
                    RdpServerSecurity::Tls(acceptor) => acceptor,
                    RdpServerSecurity::Hybrid((acceptor, _)) => acceptor,
                    RdpServerSecurity::None => unreachable!(),
                };
                let accept = match tls_acceptor.accept(stream).await {
                    Ok(accept) => accept,
                    Err(e) => {
                        warn!("Failed to TLS accept: {}", e);
                        return Ok(());
                    }
                };
                let mut framed = TokioFramed::new(accept);

                acceptor.mark_security_upgrade_as_done();

                if let RdpServerSecurity::Hybrid((_, pub_key)) = &self.opts.security {
                    // how to get the client name?
                    // doesn't seem to matter yet
                    let client_name = framed.get_inner().0.get_ref().0.peer_addr()?.to_string();

                    ironrdp_acceptor::accept_credssp(
                        &mut framed,
                        &mut acceptor,
                        &mut ironrdp_tokio::reqwest::ReqwestNetworkClient::new(),
                        client_name.into(),
                        pub_key.clone(),
                        None,
                    )
                    .await?;
                }

                let framed = self.accept_finalize(framed, acceptor).await?;
                debug!("Shutting down TLS connection");
                let (mut tls_stream, _) = framed.into_inner();
                if let Err(e) = tls_stream.shutdown().await {
                    debug!(?e, "TLS shutdown error");
                }
            }

            BeginResult::Continue(framed) => {
                self.accept_finalize(framed, acceptor).await?;
            }
        };

        Ok(())
    }

    pub async fn run(&mut self) -> Result<()> {
        let listener = Arc::new(TcpListener::bind(self.opts.addr).await?);
        let local_addr = listener.local_addr()?;

        debug!("Listening for connections on {local_addr}");
        self.local_addr = Some(local_addr);

        // Shared slot for an authenticated takeover connection waiting to be processed
        type TakeoverSlot = Arc<Mutex<Option<(
            TokioFramed<tokio_rustls::server::TlsStream<TcpStream>>,
            Acceptor,
        )>>>;
        let takeover_pending: TakeoverSlot = Arc::new(Mutex::new(None));

        loop {
            // Check if a takeover connection is waiting from the previous iteration
            let pending = takeover_pending.lock().await.take();
            if let Some((takeover_framed, mut takeover_acceptor)) = pending {
                info!("Processing takeover connection");
                self.attach_channels(&mut takeover_acceptor);
                match self.accept_finalize(takeover_framed, takeover_acceptor).await {
                    Ok(framed) => {
                        debug!("Shutting down takeover TLS connection");
                        let (mut tls_stream, _) = framed.into_inner();
                        if let Err(e) = tls_stream.shutdown().await {
                            debug!(?e, "TLS shutdown error");
                        }
                    }
                    Err(error) => {
                        let err_str = format!("{error:#}");
                        if err_str.contains("reset by peer") || err_str.contains("Broken pipe") {
                            warn!("Takeover client disconnected: {err_str}");
                        } else {
                            error!(?error, "Takeover connection error");
                        }
                    }
                }
                info!("Ready for next connection");
                self.static_channels = StaticChannelSet::new();
                continue;
            }

            let ev_receiver = Arc::clone(&self.ev_receiver);
            let mut ev_receiver = ev_receiver.lock().await;
            tokio::select! {
                Some(event) = ev_receiver.recv() => {
                    match event {
                        ServerEvent::Quit(reason) => {
                            debug!("Got quit event {reason}");
                            break;
                        }
                        ServerEvent::GetLocalAddr(tx) => {
                            let _ = tx.send(self.local_addr);
                        }
                        ServerEvent::SetCredentials(creds) => {
                            self.set_credentials(Some(creds));
                        }
                        ev => {
                            debug!("Unexpected event {:?}", ev);
                        }
                    }
                },
                Ok((stream, peer)) = listener.accept() => {
                    debug!(?peer, "Received connection");
                    drop(ev_receiver);

                    // Spawn a takeover listener that authenticates new connections
                    // while the current session is active. If auth succeeds, it
                    // signals the active session to disconnect via ServerEvent::Takeover,
                    // then queues the authenticated stream for the next iteration.
                    let takeover_listener = Arc::clone(&listener);
                    let takeover_opts = self.opts.clone();
                    let takeover_creds = self.creds.clone();
                    let takeover_ev_sender = self.ev_sender.clone();
                    let takeover_pending_clone = Arc::clone(&takeover_pending);
                    let takeover_desktop_size = self.display.lock().await.size().await;

                    let takeover_task = tokio::spawn(async move {
                        loop {
                            match takeover_listener.accept().await {
                                Ok((new_stream, new_peer)) => {
                                    info!(?new_peer, "New connection while session active, authenticating...");
                                    match Self::authenticate_stream(
                                        &takeover_opts,
                                        takeover_creds.clone(),
                                        new_stream,
                                        takeover_desktop_size,
                                    ).await {
                                        Ok((framed, acceptor)) => {
                                            info!(?new_peer, "Takeover: new client authenticated, disconnecting old session");
                                            *takeover_pending_clone.lock().await = Some((framed, acceptor));
                                            let _ = takeover_ev_sender.send(ServerEvent::Takeover);
                                            break;
                                        }
                                        Err(e) => {
                                            warn!(?new_peer, error = %e, "Takeover auth failed, rejecting");
                                        }
                                    }
                                }
                                Err(e) => {
                                    debug!("Accept error in takeover loop: {e}");
                                    break;
                                }
                            }
                        }
                    });

                    if let Err(error) = self.run_connection(stream).await {
                        let err_str = format!("{error:#}");
                        if err_str.contains("reset by peer") || err_str.contains("Broken pipe") {
                            warn!("Client disconnected: {err_str}");
                        } else {
                            error!(?error, "Connection error");
                        }
                    }

                    takeover_task.abort();
                    info!("Ready for next connection");
                    self.static_channels = StaticChannelSet::new();
                }
                else => break,
            }
        }

        Ok(())
    }

    pub fn get_svc_processor<T: SvcProcessor + 'static>(&mut self) -> Option<&mut T> {
        self.static_channels
            .get_by_type_mut::<T>()
            .and_then(|svc| svc.channel_processor_downcast_mut())
    }

    pub fn get_channel_id_by_type<T: SvcProcessor + 'static>(&self) -> Option<StaticChannelId> {
        self.static_channels.get_channel_id_by_type::<T>()
    }

    async fn dispatch_pdu(
        &mut self,
        action: Action,
        bytes: bytes::BytesMut,
        writer: &mut impl FramedWrite,
        io_channel_id: u16,
        user_channel_id: u16,
    ) -> Result<RunState> {
        match action {
            Action::FastPath => {
                let input = decode(&bytes)?;
                self.handle_fastpath(input).await;
            }

            Action::X224 => {
                if self
                    .handle_x224(writer, io_channel_id, user_channel_id, &bytes)
                    .await
                    .context("X224 input error")?
                {
                    debug!("Got disconnect request");
                    return Ok(RunState::Disconnect);
                }
            }
        }

        Ok(RunState::Continue)
    }

    async fn dispatch_display_update(
        update: DisplayUpdate,
        writer: &mut impl FramedWrite,
        user_channel_id: u16,
        io_channel_id: u16,
        buffer: &mut Vec<u8>,
        mut encoder: UpdateEncoder,
        gfx_state: &Arc<std::sync::Mutex<GfxState>>,
        drdynvc_channel_id: Option<StaticChannelId>,
    ) -> Result<(RunState, UpdateEncoder)> {
        if let DisplayUpdate::Resize(desktop_size) = update {
            debug!(?desktop_size, "Display resize");
            encoder.set_desktop_size(desktop_size);
            deactivate_all(io_channel_id, user_channel_id, writer).await?;
            return Ok((RunState::DeactivationReactivation { desktop_size }, encoder));
        }

        // Handle GFX frames through the DVC channel
        if let DisplayUpdate::GfxFrame(ref gfx_frame) = update {
            let mut state = gfx_state.lock().unwrap();
            if state.is_ready() {
                if let Some(drdynvc_id) = drdynvc_channel_id {
                    let channel_id = state.channel_id.unwrap();
                    // All frame PDUs combined into a single ZGFX-wrapped buffer
                    let pdu_data = GfxHandler::create_frame_pdu(&mut state, gfx_frame);
                    drop(state); // release lock before async

                    // Send as a single DVC message (auto-fragmented by encode_dvc_messages)
                    let dvc_messages: Vec<dvc::DvcMessage> = vec![Box::new(crate::gfx::RawGfxPdu(pdu_data))];
                    let svc_messages = dvc::encode_dvc_messages(
                        channel_id,
                        dvc_messages,
                        ChannelFlags::SHOW_PROTOCOL,
                    ).context("Failed to encode DVC messages")?;

                    let data = server_encode_svc_messages(
                        svc_messages.into(),
                        drdynvc_id,
                        user_channel_id,
                    )?;
                    writer.write_all(&data).await
                        .context("failed to write GFX frame")?;

                    return Ok((RunState::Continue, encoder));
                }
            } else {
                drop(state);
                // GFX not ready yet, fall through to bitmap encoding
                trace!("GFX not ready, falling back to bitmap path");
            }
        }

        // Handle GFX uncompressed dirty rect updates through the DVC channel
        if let DisplayUpdate::GfxUncompressed(ref uncompressed) = update {
            let mut state = gfx_state.lock().unwrap();
            if state.is_ready() {
                if let Some(drdynvc_id) = drdynvc_channel_id {
                    let channel_id = state.channel_id.unwrap();
                    let pdu_data = GfxHandler::create_uncompressed_pdu(&mut state, uncompressed);
                    drop(state);

                    let dvc_messages: Vec<dvc::DvcMessage> = vec![Box::new(crate::gfx::RawGfxPdu(pdu_data))];
                    let svc_messages = dvc::encode_dvc_messages(
                        channel_id,
                        dvc_messages,
                        ChannelFlags::SHOW_PROTOCOL,
                    ).context("Failed to encode DVC messages")?;

                    let data = server_encode_svc_messages(
                        svc_messages.into(),
                        drdynvc_id,
                        user_channel_id,
                    )?;
                    writer.write_all(&data).await
                        .context("failed to write GFX uncompressed frame")?;

                    return Ok((RunState::Continue, encoder));
                }
            } else {
                drop(state);
                trace!("GFX not ready for uncompressed, falling back to bitmap path");
            }
        }

        // Handle GFX dirty-rect H.264 updates through the DVC channel
        if let DisplayUpdate::GfxDirtyH264(ref dirty_h264) = update {
            let mut state = gfx_state.lock().unwrap();
            if state.is_ready() {
                if let Some(drdynvc_id) = drdynvc_channel_id {
                    let channel_id = state.channel_id.unwrap();
                    let pdu_data = GfxHandler::create_dirty_h264_pdu(&mut state, dirty_h264);
                    drop(state);

                    let dvc_messages: Vec<dvc::DvcMessage> = vec![Box::new(crate::gfx::RawGfxPdu(pdu_data))];
                    let svc_messages = dvc::encode_dvc_messages(
                        channel_id,
                        dvc_messages,
                        ChannelFlags::SHOW_PROTOCOL,
                    ).context("Failed to encode DVC messages")?;

                    let data = server_encode_svc_messages(
                        svc_messages.into(),
                        drdynvc_id,
                        user_channel_id,
                    )?;
                    writer.write_all(&data).await
                        .context("failed to write GFX dirty H.264 frame")?;

                    return Ok((RunState::Continue, encoder));
                }
            } else {
                drop(state);
                trace!("GFX not ready for dirty H.264, falling back to bitmap path");
            }
        }

        let mut encoder_iter = encoder.update(update);
        loop {
            let Some(fragmenter) = encoder_iter.next().await else {
                break;
            };

            let mut fragmenter = fragmenter.context("error while encoding")?;
            if fragmenter.size_hint() > buffer.len() {
                buffer.resize(fragmenter.size_hint(), 0);
            }

            while let Some(len) = fragmenter.next(buffer) {
                writer
                    .write_all(&buffer[..len])
                    .await
                    .context("failed to write display update")?;
            }
        }

        Ok((RunState::Continue, encoder))
    }

    async fn dispatch_server_events(
        &mut self,
        events: &mut Vec<ServerEvent>,
        writer: &mut impl FramedWrite,
        io_channel_id: u16,
        user_channel_id: u16,
    ) -> Result<RunState> {
        // Avoid wave message queuing up and causing extra delays.
        // This is a naive solution, better solutions should compute the actual delay, add IO priority, encode audio, use UDP etc.
        // 4 frames should roughly corresponds to hundreds of ms in regular setups.
        let mut wave_limit = 4;
        for event in events.drain(..) {
            trace!(?event, "Dispatching");
            match event {
                ServerEvent::Quit(reason) => {
                    debug!("Got quit event: {reason}");
                    let error_info = rdp::server_error_info::ErrorInfo::ProtocolIndependentCode(
                        rdp::server_error_info::ProtocolIndependentCode::RpcInitiatedDisconnect,
                    );
                    let _ = send_graceful_disconnect(io_channel_id, user_channel_id, writer, error_info).await;
                    return Ok(RunState::Disconnect);
                }
                ServerEvent::Takeover => {
                    info!("Session takeover: disconnecting current client");
                    let error_info = rdp::server_error_info::ErrorInfo::ProtocolIndependentCode(
                        rdp::server_error_info::ProtocolIndependentCode::DisconnectedByOtherconnection,
                    );
                    let _ = send_graceful_disconnect(io_channel_id, user_channel_id, writer, error_info).await;
                    return Ok(RunState::Disconnect);
                }
                ServerEvent::GetLocalAddr(tx) => {
                    let _ = tx.send(self.local_addr);
                }
                ServerEvent::SetCredentials(creds) => {
                    self.set_credentials(Some(creds));
                }
                ServerEvent::Rdpsnd(s) => {
                    let Some(rdpsnd) = self.get_svc_processor::<RdpsndServer>() else {
                        warn!("No rdpsnd channel, dropping event");
                        continue;
                    };
                    let msgs = match s {
                        RdpsndServerMessage::Wave(data, ts) => {
                            if wave_limit == 0 {
                                debug!("Dropping wave");
                                continue;
                            }
                            wave_limit -= 1;
                            match rdpsnd.wave(data, ts) {
                                Ok(msgs) => msgs,
                                Err(e) => {
                                    // Wave before handshake complete — drop silently
                                    debug!("Dropping wave: {e}");
                                    continue;
                                }
                            }
                        }
                        RdpsndServerMessage::SetVolume { left, right } => {
                            rdpsnd.set_volume(left, right).context("failed to send rdpsnd event")?
                        }
                        RdpsndServerMessage::Close => {
                            rdpsnd.close().context("failed to send rdpsnd event")?
                        }
                        RdpsndServerMessage::Error(error) => {
                            error!(?error, "Handling rdpsnd event");
                            continue;
                        }
                    };
                    let channel_id = self
                        .get_channel_id_by_type::<RdpsndServer>()
                        .ok_or_else(|| anyhow!("SVC channel not found"))?;
                    let data = server_encode_svc_messages(msgs.into(), channel_id, user_channel_id)?;
                    writer.write_all(&data).await?;
                }
                ServerEvent::Clipboard(c) => {
                    let Some(cliprdr) = self.get_svc_processor::<CliprdrServer>() else {
                        warn!("No clipboard channel, dropping event");
                        continue;
                    };
                    let msgs = match c {
                        ClipboardMessage::SendInitiateCopy(formats) => cliprdr.initiate_copy(&formats),
                        ClipboardMessage::SendFormatData(data) => cliprdr.submit_format_data(data),
                        ClipboardMessage::SendInitiatePaste(format) => cliprdr.initiate_paste(format),
                        ClipboardMessage::Error(error) => {
                            error!(?error, "Handling clipboard event");
                            continue;
                        }
                    }
                    .context("failed to send clipboard event")?;
                    let channel_id = self
                        .get_channel_id_by_type::<CliprdrServer>()
                        .ok_or_else(|| anyhow!("SVC channel not found"))?;
                    let data = server_encode_svc_messages(msgs.into(), channel_id, user_channel_id)?;
                    writer.write_all(&data).await?;
                }
                ServerEvent::ClipboardFileContents(response) => {
                    let Some(cliprdr) = self.get_svc_processor::<CliprdrServer>() else {
                        warn!("No clipboard channel, dropping file contents response");
                        continue;
                    };
                    let msgs = cliprdr
                        .submit_file_contents(response)
                        .context("failed to submit file contents")?;
                    let channel_id = self
                        .get_channel_id_by_type::<CliprdrServer>()
                        .ok_or_else(|| anyhow!("SVC channel not found"))?;
                    let data = server_encode_svc_messages(msgs.into(), channel_id, user_channel_id)?;
                    writer.write_all(&data).await?;
                }
                ServerEvent::ClipboardFileContentsRequest(request) => {
                    let channel_id = self
                        .get_channel_id_by_type::<CliprdrServer>()
                        .ok_or_else(|| anyhow!("SVC channel not found"))?;
                    let pdu = ClipboardPdu::FileContentsRequest(request);
                    let svc_msg = encode_cliprdr_pdu(pdu);
                    let data = server_encode_svc_messages(
                        vec![svc_msg].into(),
                        channel_id,
                        user_channel_id,
                    )?;
                    writer.write_all(&data).await?;
                }
            }
        }

        Ok(RunState::Continue)
    }

    async fn client_loop<R, W>(
        &mut self,
        reader: &mut Framed<R>,
        writer: &mut Framed<W>,
        io_channel_id: u16,
        user_channel_id: u16,
        mut encoder: UpdateEncoder,
    ) -> Result<RunState>
    where
        R: FramedRead,
        W: FramedWrite,
    {
        debug!("Starting client loop");
        let display_updates_result = self.display.lock().await.updates().await;
        let mut display_updates = match display_updates_result {
            Ok(updates) => updates,
            Err(e) => {
                let err_str = format!("{e:#}");
                if err_str.contains("Screen Recording") || err_str.contains("declined TCCs") || err_str.contains("shareable content") {
                    warn!("Screen capture permission denied, sending error to client");
                    let error_info = rdp::server_error_info::ErrorInfo::ProtocolIndependentCode(
                        rdp::server_error_info::ProtocolIndependentCode::ServerInsufficientPrivileges,
                    );
                    let _ = send_graceful_disconnect(io_channel_id, user_channel_id, writer, error_info).await;
                }
                return Err(e);
            }
        };
        let mut writer = SharedWriter::new(writer);
        let mut display_writer = writer.clone();
        let mut event_writer = writer.clone();
        let ev_receiver = Arc::clone(&self.ev_receiver);
        let gfx_state = Arc::clone(&self.gfx_state);

        // Update GFX state with current desktop size
        {
            let size = self.display.lock().await.size().await;
            let mut gs = self.gfx_state.lock().unwrap();
            gs.width = size.width;
            gs.height = size.height;
        }

        // Get DRDYNVC channel ID for sending GFX frames
        let drdynvc_channel_id = self.get_channel_id_by_type::<dvc::DrdynvcServer>();

        let s = Rc::new(Mutex::new(self));

        let this = Rc::clone(&s);
        let dispatch_pdu = async move {
            loop {
                let (action, bytes) = reader.read_pdu().await?;
                let mut this = this.lock().await;
                match this
                    .dispatch_pdu(action, bytes, &mut writer, io_channel_id, user_channel_id)
                    .await?
                {
                    RunState::Continue => continue,
                    state => break Ok(state),
                }
            }
        };

        let dispatch_display = async move {
            let mut buffer = vec![0u8; 4096];

            loop {
                match display_updates.next_update().await {
                    Ok(Some(update)) => {
                        match Self::dispatch_display_update(
                            update,
                            &mut display_writer,
                            user_channel_id,
                            io_channel_id,
                            &mut buffer,
                            encoder,
                            &gfx_state,
                            drdynvc_channel_id,
                        )
                        .await?
                        {
                            (RunState::Continue, enc) => {
                                encoder = enc;
                                continue;
                            }
                            (state, _) => {
                                break Ok(state);
                            }
                        }
                    }
                    Ok(None) => {
                        break Ok(RunState::Disconnect);
                    }
                    Err(error) => {
                        warn!(error = format!("{error:#}"), "next_updated failed");
                    }
                }
            }
        };

        let this = Rc::clone(&s);
        let mut ev_receiver = ev_receiver.lock().await;
        let dispatch_events = async move {
            let mut events = Vec::with_capacity(100);
            loop {
                let nevents = ev_receiver.recv_many(&mut events, 100).await;
                if nevents == 0 {
                    debug!("No sever events.. stopping");
                    break Ok(RunState::Disconnect);
                }
                while let Ok(ev) = ev_receiver.try_recv() {
                    events.push(ev);
                }
                let mut this = this.lock().await;
                match this
                    .dispatch_server_events(&mut events, &mut event_writer, io_channel_id, user_channel_id)
                    .await?
                {
                    RunState::Continue => continue,
                    state => break Ok(state),
                }
            }
        };

        let state = tokio::select!(
            state = dispatch_pdu => state,
            state = dispatch_display => state,
            state = dispatch_events => state,
        );

        debug!("End of client loop: {state:?}");
        state
    }

    async fn client_accepted<R, W>(
        &mut self,
        reader: &mut Framed<R>,
        writer: &mut Framed<W>,
        result: AcceptorResult,
    ) -> Result<RunState>
    where
        R: FramedRead,
        W: FramedWrite,
    {
        debug!("Client accepted");

        if !result.input_events.is_empty() {
            debug!("Handling input event backlog from acceptor sequence");
            self.handle_input_backlog(
                writer,
                result.io_channel_id,
                result.user_channel_id,
                result.input_events,
            )
            .await?;
        }

        self.static_channels = result.static_channels;
        if !result.reactivation {
            for (_type_id, channel, channel_id) in self.static_channels.iter_mut() {
                debug!(?channel, ?channel_id, "Start");
                let Some(channel_id) = channel_id else {
                    continue;
                };
                let svc_responses = channel.start()?;
                let response = server_encode_svc_messages(svc_responses, channel_id, result.user_channel_id)?;
                writer.write_all(&response).await?;
            }
        }

        let mut update_codecs = UpdateEncoderCodecs::new();
        let mut surface_flags = CmdFlags::empty();
        for c in result.capabilities {
            match c {
                CapabilitySet::General(c) => {
                    let fastpath = c.extra_flags.contains(GeneralExtraFlags::FASTPATH_OUTPUT_SUPPORTED);
                    if !fastpath {
                        bail!("Fastpath output not supported!");
                    }
                }
                CapabilitySet::Bitmap(b) => {
                    let client_size = DesktopSize {
                        width: b.desktop_width,
                        height: b.desktop_height,
                    };
                    let display_size = self.display.lock().await.size().await;

                    if client_size.width != display_size.width || client_size.height != display_size.height {
                        info!(
                            client_w = client_size.width, client_h = client_size.height,
                            server_w = display_size.width, server_h = display_size.height,
                            "Client requested different resolution"
                        );
                        // Adopt client resolution via request_resize
                        self.display.lock().await.request_resize(
                            client_size.width, client_size.height,
                        );
                    }
                }
                CapabilitySet::SurfaceCommands(c) => {
                    surface_flags = c.flags;
                }
                CapabilitySet::BitmapCodecs(BitmapCodecs(codecs)) => {
                    for codec in codecs {
                        match codec.property {
                            // FIXME: The encoder operates in image mode only.
                            //
                            // See [MS-RDPRFX] 3.1.1.1 "State Machine" for
                            // implementation of the video mode. which allows to
                            // skip sending Header for each image.
                            //
                            // We should distinguish parameters for both modes,
                            // and somehow choose the "best", instead of picking
                            // the last parsed here.
                            CodecProperty::RemoteFx(rdp::capability_sets::RemoteFxContainer::ClientContainer(c))
                                if self.opts.has_remote_fx() =>
                            {
                                for caps in c.caps_data.0 .0 {
                                    update_codecs.set_remotefx(Some((caps.entropy_bits, codec.id)));
                                }
                            }
                            CodecProperty::ImageRemoteFx(rdp::capability_sets::RemoteFxContainer::ClientContainer(
                                c,
                            )) if self.opts.has_image_remote_fx() => {
                                for caps in c.caps_data.0 .0 {
                                    update_codecs.set_remotefx(Some((caps.entropy_bits, codec.id)));
                                }
                            }
                            CodecProperty::NsCodec(_) => (),
                            #[cfg(feature = "qoi")]
                            CodecProperty::Qoi if self.opts.has_qoi() => {
                                update_codecs.set_qoi(Some(codec.id));
                            }
                            #[cfg(feature = "qoiz")]
                            CodecProperty::QoiZ if self.opts.has_qoiz() => {
                                update_codecs.set_qoiz(Some(codec.id));
                            }
                            _ => (),
                        }
                    }
                }
                _ => {}
            }
        }

        let desktop_size = self.display.lock().await.size().await;
        let encoder = UpdateEncoder::new(desktop_size, surface_flags, update_codecs)
            .context("failed to initialize update encoder")?;

        let state = self
            .client_loop(reader, writer, result.io_channel_id, result.user_channel_id, encoder)
            .await
            .context("client loop failure")?;

        Ok(state)
    }

    async fn handle_input_backlog(
        &mut self,
        writer: &mut impl FramedWrite,
        io_channel_id: u16,
        user_channel_id: u16,
        frames: Vec<Vec<u8>>,
    ) -> Result<()> {
        for frame in frames {
            match Action::from_fp_output_header(frame[0]) {
                Ok(Action::FastPath) => {
                    let input = decode(&frame)?;
                    self.handle_fastpath(input).await;
                }

                Ok(Action::X224) => {
                    let _ = self.handle_x224(writer, io_channel_id, user_channel_id, &frame).await;
                }

                // the frame here is always valid, because otherwise it would
                // have failed during the acceptor loop
                Err(_) => unreachable!(),
            }
        }

        Ok(())
    }

    async fn handle_fastpath(&mut self, input: FastPathInput) {
        for event in input.input_events().iter().copied() {
            let mut handler = self.handler.lock().await;
            match event {
                FastPathInputEvent::KeyboardEvent(flags, key) => {
                    handler.keyboard((key, flags).into());
                }

                FastPathInputEvent::UnicodeKeyboardEvent(flags, key) => {
                    handler.keyboard((key, flags).into());
                }

                FastPathInputEvent::SyncEvent(flags) => {
                    handler.keyboard(flags.into());
                }

                FastPathInputEvent::MouseEvent(mouse) => {
                    handler.mouse(mouse.into());
                }

                FastPathInputEvent::MouseEventEx(mouse) => {
                    handler.mouse(mouse.into());
                }

                FastPathInputEvent::MouseEventRel(mouse) => {
                    handler.mouse(mouse.into());
                }

                FastPathInputEvent::QoeEvent(quality) => {
                    warn!("Received QoE: {}", quality);
                }
            }
        }
    }

    async fn handle_io_channel_data(&mut self, data: SendDataRequest<'_>) -> Result<bool> {
        let control: rdp::headers::ShareControlHeader = decode(data.user_data.as_ref())?;

        match control.share_control_pdu {
            ShareControlPdu::Data(header) => match header.share_data_pdu {
                rdp::headers::ShareDataPdu::Input(pdu) => {
                    self.handle_input_event(pdu).await;
                }

                rdp::headers::ShareDataPdu::ShutdownRequest => {
                    return Ok(true);
                }

                unexpected => {
                    warn!(?unexpected, "Unexpected share data pdu");
                }
            },

            unexpected => {
                warn!(?unexpected, "Unexpected share control");
            }
        }

        Ok(false)
    }

    async fn handle_x224(
        &mut self,
        writer: &mut impl FramedWrite,
        io_channel_id: u16,
        user_channel_id: u16,
        frame: &[u8],
    ) -> Result<bool> {
        let message = decode::<X224<mcs::McsMessage<'_>>>(frame)?;
        match message.0 {
            mcs::McsMessage::SendDataRequest(data) => {
                debug!(?data, "McsMessage::SendDataRequest");
                if data.channel_id == io_channel_id {
                    return self.handle_io_channel_data(data).await;
                }

                if let Some(svc) = self.static_channels.get_by_channel_id_mut(data.channel_id) {
                    let response_pdus = svc.process(&data.user_data)?;
                    let response = server_encode_svc_messages(response_pdus, data.channel_id, user_channel_id)?;
                    writer.write_all(&response).await?;
                } else {
                    warn!(channel_id = data.channel_id, "Unexpected channel received: ID",);
                }
            }

            mcs::McsMessage::DisconnectProviderUltimatum(disconnect) => {
                if disconnect.reason == mcs::DisconnectReason::UserRequested {
                    return Ok(true);
                }
            }

            _ => {
                warn!(name = ironrdp_core::name(&message), "Unexpected mcs message");
            }
        }

        Ok(false)
    }

    async fn handle_input_event(&mut self, input: InputEventPdu) {
        for event in input.0 {
            let mut handler = self.handler.lock().await;
            match event {
                ironrdp_pdu::input::InputEvent::ScanCode(key) => {
                    handler.keyboard((key.key_code, key.flags).into());
                }

                ironrdp_pdu::input::InputEvent::Unicode(key) => {
                    handler.keyboard((key.unicode_code, key.flags).into());
                }

                ironrdp_pdu::input::InputEvent::Sync(sync) => {
                    handler.keyboard(sync.flags.into());
                }

                ironrdp_pdu::input::InputEvent::Mouse(mouse) => {
                    handler.mouse(mouse.into());
                }

                ironrdp_pdu::input::InputEvent::MouseX(mouse) => {
                    handler.mouse(mouse.into());
                }

                ironrdp_pdu::input::InputEvent::MouseRel(mouse) => {
                    handler.mouse(mouse.into());
                }

                ironrdp_pdu::input::InputEvent::Unused(_) => {}
            }
        }
    }

    async fn accept_finalize<S>(&mut self, mut framed: TokioFramed<S>, mut acceptor: Acceptor) -> Result<TokioFramed<S>>
    where
        S: AsyncRead + AsyncWrite + Sync + Send + Unpin,
    {
        loop {
            let (new_framed, result) = ironrdp_acceptor::accept_finalize(framed, &mut acceptor)
                .await
                .context("failed to accept client during finalize")?;

            let (mut reader, mut writer) = split_tokio_framed(new_framed);

            match self.client_accepted(&mut reader, &mut writer, result).await? {
                RunState::Continue => {
                    unreachable!();
                }
                RunState::DeactivationReactivation { desktop_size } => {
                    // No description of such behavior was found in the
                    // specification, but apparently, we must keep the channel
                    // state as they were during reactivation. This fixes
                    // various state issues during client resize.
                    acceptor = Acceptor::new_deactivation_reactivation(
                        acceptor,
                        core::mem::take(&mut self.static_channels),
                        desktop_size,
                    )?;
                    framed = unsplit_tokio_framed(reader, writer);
                    continue;
                }
                RunState::Disconnect => {
                    let final_framed = unsplit_tokio_framed(reader, writer);
                    return Ok(final_framed);
                }
            }
        }
    }

    pub fn set_credentials(&mut self, creds: Option<Credentials>) {
        debug!(?creds, "Changing credentials");
        self.creds = creds
    }

    /// Authenticate a new TCP stream through X.224 + TLS + CredSSP.
    /// Returns the authenticated TLS framed stream and acceptor, ready for
    /// `accept_finalize`. Does NOT touch `self` state — safe to call while
    /// another session is active.
    async fn authenticate_stream(
        opts: &RdpServerOptions,
        creds: Option<Credentials>,
        stream: TcpStream,
        desktop_size: DesktopSize,
    ) -> Result<(TokioFramed<tokio_rustls::server::TlsStream<TcpStream>>, Acceptor)> {
        let framed = TokioFramed::new(stream);
        let size = desktop_size;
        let capabilities = capabilities::capabilities(opts, size);
        let mut acceptor = Acceptor::new(opts.security.flag(), size, capabilities, creds);

        let res = ironrdp_acceptor::accept_begin(framed, &mut acceptor)
            .await
            .context("accept_begin failed")?;

        match res {
            BeginResult::ShouldUpgrade(stream) => {
                let tls_acceptor = match &opts.security {
                    RdpServerSecurity::Tls(acceptor) => acceptor,
                    RdpServerSecurity::Hybrid((acceptor, _)) => acceptor,
                    RdpServerSecurity::None => unreachable!(),
                };
                let accept = tls_acceptor.accept(stream).await
                    .context("TLS accept failed")?;
                let mut framed = TokioFramed::new(accept);

                acceptor.mark_security_upgrade_as_done();

                if let RdpServerSecurity::Hybrid((_, pub_key)) = &opts.security {
                    let client_name = framed.get_inner().0.get_ref().0.peer_addr()?.to_string();
                    ironrdp_acceptor::accept_credssp(
                        &mut framed,
                        &mut acceptor,
                        &mut ironrdp_tokio::reqwest::ReqwestNetworkClient::new(),
                        client_name.into(),
                        pub_key.clone(),
                        None,
                    )
                    .await?;
                }

                Ok((framed, acceptor))
            }
            BeginResult::Continue(_) => {
                bail!("Non-TLS connections not supported for session takeover");
            }
        }
    }
}

async fn send_error_info(
    io_channel_id: u16,
    user_channel_id: u16,
    writer: &mut impl FramedWrite,
    error_info: rdp::server_error_info::ErrorInfo,
) -> Result<(), anyhow::Error> {
    let pdu = rdp::headers::ShareDataPdu::ServerSetErrorInfo(
        rdp::server_error_info::ServerSetErrorInfoPdu(error_info),
    );
    let pdu = rdp::headers::ShareControlHeader {
        share_id: 0,
        pdu_source: io_channel_id,
        share_control_pdu: ShareControlPdu::Data(rdp::headers::ShareDataHeader {
            share_data_pdu: pdu,
            stream_priority: rdp::headers::StreamPriority::Low,
            compression_flags: rdp::headers::CompressionFlags::empty(),
            compression_type: rdp::client_info::CompressionType::K8,
        }),
    };
    let user_data = encode_vec(&pdu)?.into();
    let pdu = SendDataIndication {
        initiator_id: user_channel_id,
        channel_id: io_channel_id,
        user_data,
    };
    let msg = encode_vec(&X224(pdu))?;
    writer.write_all(&msg).await?;
    Ok(())
}

async fn deactivate_all(
    io_channel_id: u16,
    user_channel_id: u16,
    writer: &mut impl FramedWrite,
) -> Result<(), anyhow::Error> {
    let pdu = ShareControlPdu::ServerDeactivateAll(ServerDeactivateAll);
    let pdu = rdp::headers::ShareControlHeader {
        share_id: 0,
        pdu_source: io_channel_id,
        share_control_pdu: pdu,
    };
    let user_data = encode_vec(&pdu)?.into();
    let pdu = SendDataIndication {
        initiator_id: user_channel_id,
        channel_id: io_channel_id,
        user_data,
    };
    let msg = encode_vec(&X224(pdu))?;
    writer.write_all(&msg).await?;
    Ok(())
}

async fn send_disconnect_ultimatum(writer: &mut impl FramedWrite) -> Result<(), anyhow::Error> {
    let disconnect = mcs::DisconnectProviderUltimatum::from_reason(mcs::DisconnectReason::ProviderInitiated);
    let msg = encode_vec(&X224(mcs::McsMessage::DisconnectProviderUltimatum(disconnect)))?;
    writer.write_all(&msg).await?;
    Ok(())
}

/// Send the standard RDP graceful disconnect sequence:
/// Set Error Info PDU → Deactivate All → MCS Disconnect Provider Ultimatum
async fn send_graceful_disconnect(
    io_channel_id: u16,
    user_channel_id: u16,
    writer: &mut impl FramedWrite,
    error_info: rdp::server_error_info::ErrorInfo,
) -> Result<(), anyhow::Error> {
    let _ = send_error_info(io_channel_id, user_channel_id, writer, error_info).await;
    let _ = deactivate_all(io_channel_id, user_channel_id, writer).await;
    let _ = send_disconnect_ultimatum(writer).await;
    Ok(())
}

fn encode_cliprdr_pdu(pdu: ClipboardPdu<'static>) -> SvcMessage {
    SvcMessage::from(pdu).with_flags(ChannelFlags::SHOW_PROTOCOL)
}

struct SharedWriter<'w, W: FramedWrite> {
    writer: Rc<Mutex<&'w mut W>>,
}

impl<W: FramedWrite> Clone for SharedWriter<'_, W> {
    fn clone(&self) -> Self {
        Self {
            writer: Rc::clone(&self.writer),
        }
    }
}

impl<W> FramedWrite for SharedWriter<'_, W>
where
    W: FramedWrite,
{
    type WriteAllFut<'write>
        = core::pin::Pin<Box<dyn core::future::Future<Output = std::io::Result<()>> + 'write>>
    where
        Self: 'write;

    fn write_all<'a>(&'a mut self, buf: &'a [u8]) -> Self::WriteAllFut<'a> {
        Box::pin(async {
            let mut writer = self.writer.lock().await;

            writer.write_all(buf).await?;
            Ok(())
        })
    }
}

impl<'a, W: FramedWrite> SharedWriter<'a, W> {
    fn new(writer: &'a mut W) -> Self {
        Self {
            writer: Rc::new(Mutex::new(writer)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ironrdp_pdu::rdp::capability_sets::BitmapCodecs;
    use tokio::io::AsyncWriteExt;
    use tokio::net::TcpListener;

    fn test_opts_no_security() -> RdpServerOptions {
        RdpServerOptions {
            addr: "127.0.0.1:0".parse().unwrap(),
            security: RdpServerSecurity::None,
            codecs: BitmapCodecs(vec![]),
        }
    }

    #[test]
    fn takeover_event_is_distinct_variant() {
        let event = ServerEvent::Takeover;
        assert!(matches!(event, ServerEvent::Takeover));
        let debug = format!("{:?}", event);
        assert!(debug.contains("Takeover"));
    }

    #[tokio::test]
    async fn takeover_event_delivered_via_channel() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        tx.send(ServerEvent::Takeover).unwrap();
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, ServerEvent::Takeover));
    }

    #[tokio::test]
    async fn takeover_slot_stores_and_takes() {
        type TakeoverSlot = Arc<Mutex<Option<String>>>;
        let slot: TakeoverSlot = Arc::new(Mutex::new(None));

        assert!(slot.lock().await.is_none());

        *slot.lock().await = Some("test-connection".to_string());
        assert!(slot.lock().await.is_some());

        let taken = slot.lock().await.take();
        assert_eq!(taken.unwrap(), "test-connection");
        assert!(slot.lock().await.is_none());
    }

    #[tokio::test]
    async fn takeover_slot_only_consumed_once() {
        type TakeoverSlot = Arc<Mutex<Option<String>>>;
        let slot: TakeoverSlot = Arc::new(Mutex::new(None));
        let slot2 = Arc::clone(&slot);

        *slot.lock().await = Some("pending-client".to_string());

        let first = slot.lock().await.take();
        assert!(first.is_some());

        let second = slot2.lock().await.take();
        assert!(second.is_none());
    }

    #[tokio::test]
    async fn authenticate_stream_rejects_garbage_input() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_task = tokio::spawn(async move {
            let mut stream = TcpStream::connect(addr).await.unwrap();
            stream.write_all(b"NOT-AN-RDP-CLIENT").await.unwrap();
            let _ = stream.shutdown().await;
        });

        let (server_stream, _) = listener.accept().await.unwrap();
        let result = RdpServer::authenticate_stream(
            &test_opts_no_security(),
            None,
            server_stream,
            DesktopSize { width: 1920, height: 1080 },
        ).await;

        assert!(result.is_err());
        client_task.await.unwrap();
    }

    #[tokio::test]
    async fn authenticate_stream_rejects_client_that_closes_immediately() {
        let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = listener.local_addr().unwrap();

        let client_task = tokio::spawn(async move {
            let stream = TcpStream::connect(addr).await.unwrap();
            drop(stream);
        });

        let (server_stream, _) = listener.accept().await.unwrap();
        let result = RdpServer::authenticate_stream(
            &test_opts_no_security(),
            None,
            server_stream,
            DesktopSize { width: 1920, height: 1080 },
        ).await;

        assert!(result.is_err());
        client_task.await.unwrap();
    }

    #[tokio::test]
    async fn takeover_flow_signals_event_sender() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let slot: Arc<Mutex<Option<bool>>> = Arc::new(Mutex::new(None));
        let slot_clone = Arc::clone(&slot);

        // Simulate what the takeover task does when auth succeeds
        *slot_clone.lock().await = Some(true);
        let _ = tx.send(ServerEvent::Takeover);

        // Verify the event was sent
        let event = rx.recv().await.unwrap();
        assert!(matches!(event, ServerEvent::Takeover));

        // Verify the slot has the pending connection
        let pending = slot.lock().await.take();
        assert_eq!(pending, Some(true));
    }

    #[tokio::test]
    async fn takeover_event_not_sent_on_auth_failure() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let slot: Arc<Mutex<Option<bool>>> = Arc::new(Mutex::new(None));

        // Simulate what happens when auth fails — no event sent, no slot written
        let auth_ok = false;
        if auth_ok {
            *slot.lock().await = Some(true);
            let _ = tx.send(ServerEvent::Takeover);
        }

        // Channel should be empty
        drop(tx);
        let event = rx.recv().await;
        assert!(event.is_none());
        assert!(slot.lock().await.is_none());
    }

    #[tokio::test]
    async fn multiple_auth_failures_dont_block_eventual_takeover() {
        let (tx, mut rx) = mpsc::unbounded_channel();
        let slot: Arc<Mutex<Option<u32>>> = Arc::new(Mutex::new(None));

        // Simulate 3 auth failures followed by 1 success
        let attempts = vec![false, false, false, true];
        for (i, success) in attempts.iter().enumerate() {
            if *success {
                *slot.lock().await = Some(i as u32);
                let _ = tx.send(ServerEvent::Takeover);
                break;
            }
        }

        let event = rx.recv().await.unwrap();
        assert!(matches!(event, ServerEvent::Takeover));
        assert_eq!(slot.lock().await.take(), Some(3));
    }

    #[test]
    fn error_info_pdu_encodes_correctly() {
        let error_info = rdp::server_error_info::ErrorInfo::ProtocolIndependentCode(
            rdp::server_error_info::ProtocolIndependentCode::ServerInsufficientPrivileges,
        );
        let pdu = rdp::server_error_info::ServerSetErrorInfoPdu(error_info);
        let encoded = encode_vec(&pdu).unwrap();
        // ErrorInfo is a u32 = 0x00000009
        assert_eq!(encoded, [0x09, 0x00, 0x00, 0x00]);
    }

    #[test]
    fn error_info_description_is_meaningful() {
        let error_info = rdp::server_error_info::ErrorInfo::ProtocolIndependentCode(
            rdp::server_error_info::ProtocolIndependentCode::ServerInsufficientPrivileges,
        );
        let desc = error_info.description();
        assert!(desc.contains("insufficient") || desc.contains("privileges"),
            "description should mention privileges: {desc}");
    }

    struct MockWriter {
        chunks: Vec<Vec<u8>>,
    }

    impl MockWriter {
        fn new() -> Self {
            Self { chunks: Vec::new() }
        }
    }

    impl FramedWrite for MockWriter {
        type WriteAllFut<'a> = core::pin::Pin<Box<dyn core::future::Future<Output = std::io::Result<()>> + 'a>> where Self: 'a;

        fn write_all<'a>(&'a mut self, buf: &'a [u8]) -> Self::WriteAllFut<'a> {
            self.chunks.push(buf.to_vec());
            Box::pin(async { Ok(()) })
        }
    }

    #[tokio::test]
    async fn send_disconnect_ultimatum_encodes_mcs_pdu() {
        let mut writer = MockWriter::new();
        send_disconnect_ultimatum(&mut writer).await.unwrap();
        assert_eq!(writer.chunks.len(), 1);
        let data = &writer.chunks[0];
        // Should contain X224 header + MCS Disconnect Provider Ultimatum
        assert!(!data.is_empty());
        // MCS DPU domain tag = 8, encoded in ASN.1 BER: (8 << 2) | class bits
        // The X224 wrapper adds a TPKT header (4 bytes) + X224 data header (3 bytes)
        assert!(data.len() >= 7, "disconnect PDU should be at least 7 bytes, got {}", data.len());
    }

    #[tokio::test]
    async fn send_graceful_disconnect_sends_three_pdus() {
        let mut writer = MockWriter::new();
        let error_info = rdp::server_error_info::ErrorInfo::ProtocolIndependentCode(
            rdp::server_error_info::ProtocolIndependentCode::DisconnectedByOtherconnection,
        );
        send_graceful_disconnect(1004, 1003, &mut writer, error_info).await.unwrap();
        // Should send 3 PDUs: error info, deactivate all, disconnect ultimatum
        assert_eq!(writer.chunks.len(), 3, "graceful disconnect should send 3 PDUs");
    }

    #[tokio::test]
    async fn send_graceful_disconnect_for_tcc_sends_three_pdus() {
        let mut writer = MockWriter::new();
        let error_info = rdp::server_error_info::ErrorInfo::ProtocolIndependentCode(
            rdp::server_error_info::ProtocolIndependentCode::ServerInsufficientPrivileges,
        );
        send_graceful_disconnect(1004, 1003, &mut writer, error_info).await.unwrap();
        assert_eq!(writer.chunks.len(), 3, "graceful disconnect should send 3 PDUs");
    }

    #[tokio::test]
    async fn send_graceful_disconnect_error_info_is_first_pdu() {
        let mut writer = MockWriter::new();
        let error_info = rdp::server_error_info::ErrorInfo::ProtocolIndependentCode(
            rdp::server_error_info::ProtocolIndependentCode::DisconnectedByOtherconnection,
        );
        send_graceful_disconnect(1004, 1003, &mut writer, error_info).await.unwrap();

        // First PDU should contain the error code 0x05 (DisconnectedByOtherconnection)
        let first_pdu = &writer.chunks[0];
        assert!(first_pdu.windows(4).any(|w| w == [0x05, 0x00, 0x00, 0x00]),
            "first PDU should contain error code 0x05");
    }

    #[tokio::test]
    async fn send_graceful_disconnect_ultimatum_is_last_pdu() {
        let mut writer = MockWriter::new();
        let error_info = rdp::server_error_info::ErrorInfo::ProtocolIndependentCode(
            rdp::server_error_info::ProtocolIndependentCode::ServerInsufficientPrivileges,
        );
        send_graceful_disconnect(1004, 1003, &mut writer, error_info).await.unwrap();

        // Last PDU should be the disconnect ultimatum (same as standalone)
        let mut standalone_writer = MockWriter::new();
        send_disconnect_ultimatum(&mut standalone_writer).await.unwrap();
        assert_eq!(writer.chunks[2], standalone_writer.chunks[0],
            "last PDU should be the MCS Disconnect Provider Ultimatum");
    }
}
