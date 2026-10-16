mod config;
mod display;
mod gfx;
mod handler;
mod perf_stats;
mod tls;
mod version;

use anyhow::{Context, Result};
use clap::Parser;
use config::{Cli, ServerConfig};
use display::MacDisplay;
use handler::MacInputHandler;
use ironrdp_server::{ConnectionHandler, ConnectionPolicy, Credentials, RdpServer, TlsIdentityCtx};
use std::net::SocketAddr;

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = ServerConfig::load(&cli)?;

    // Initialize logging — write to both stderr and file
    let log_level = config.log_level.as_deref().unwrap_or("info");
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level));

    let log_path = std::env::current_dir()
        .unwrap_or_default()
        .join("macrdp.log");
    let log_file = std::fs::File::create(&log_path)?;

    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer().with_writer(std::io::stderr))
        .with(
            tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_writer(std::sync::Mutex::new(log_file)),
        )
        .init();

    eprintln!("Log file: {}", log_path.display());

    tracing::info!(
        version = version::version(),
        ?config,
        "macrdp server starting"
    );

    // Ensure TLS certificates exist
    let (cert_path, key_path) =
        tls::ensure_tls_files(config.cert_path.as_deref(), config.key_path.as_deref())?;

    // Load TLS identity
    let tls_identity = TlsIdentityCtx::init_from_paths(&cert_path, &key_path)
        .context("Failed to load TLS certificate")?;
    let tls_acceptor = tls_identity
        .make_acceptor()
        .context("Failed to create TLS acceptor")?;

    // Check and request required macOS permissions
    check_permissions();

    // Detect macOS display sizes:
    // - SCK logical: from ScreenCaptureKit SCDisplay, used for capture sizing
    // - CG logical: from CoreGraphics CGDisplay.bounds(), used for mouse mapping
    //   CGEvent operates in the CG coordinate space, so mouse mapping MUST use CG dimensions.
    let (sck_w, sck_h) = match macrdp_capture::detect_display_size() {
        Ok((w, h)) => {
            tracing::info!(width = w, height = h, "SCK logical display size");
            (w as u16, h as u16)
        }
        Err(e) => {
            tracing::warn!("Failed to detect SCK display size: {e}, defaulting to 1920x1080");
            (1920u16, 1080u16)
        }
    };
    let (cg_w, cg_h) = match macrdp_capture::detect_cg_display_size() {
        Ok((w, h)) => {
            tracing::info!(
                width = w,
                height = h,
                "CG logical display size (CGEvent coordinate space)"
            );
            (w as u16, h as u16)
        }
        Err(e) => {
            tracing::warn!("Failed to detect CG display size: {e}, falling back to SCK size");
            (sck_w, sck_h)
        }
    };
    if sck_w != cg_w || sck_h != cg_h {
        tracing::warn!(
            sck_w,
            sck_h,
            cg_w,
            cg_h,
            "SCK and CG display sizes differ! Using CG for mouse mapping, SCK for capture."
        );
    }

    // Determine RDP desktop resolution (what we capture and send)
    let (width, height) = if config.width > 0 && config.height > 0 {
        (config.width as u16, config.height as u16)
    } else {
        (sck_w, sck_h)
    };

    // Parse quality setting
    let quality = match config.quality.as_deref() {
        Some("low_latency") => macrdp_encode::Quality::LowLatency,
        Some("balanced") => macrdp_encode::Quality::Balanced,
        _ => macrdp_encode::Quality::HighQuality, // default: best quality
    };

    // Parse encoder preference
    let encoder_pref = macrdp_encode::EncoderPreference::from_str_opt(config.encoder.as_deref());
    tracing::info!(?encoder_pref, "Encoder preference");

    // Parse chroma mode
    let mode_444 = config.chroma_mode.as_deref() == Some("avc444");
    tracing::info!(
        chroma_mode = config.chroma_mode.as_deref().unwrap_or("avc420"),
        mode_444,
        "Chroma mode"
    );

    // Resolution: "auto", "WxH" (e.g. "3840x2160"), or legacy scale (e.g. "2")
    let res_mode = config.resolution.as_deref().unwrap_or("auto");
    let res_auto = res_mode == "auto";
    let (width, height) = if res_mode == "auto" {
        let scale = macrdp_capture::detect_display_scale().unwrap_or(1);
        tracing::info!(scale, width, height, "Resolution auto: display scale");
        (width * scale as u16, height * scale as u16)
    } else if let Some((w, h)) = res_mode
        .split_once('x')
        .and_then(|(w, h)| Some((w.parse::<u16>().ok()?, h.parse::<u16>().ok()?)))
    {
        tracing::info!(w, h, "Resolution: explicit WxH");
        (w, h)
    } else if let Ok(scale) = res_mode.parse::<u32>() {
        let scale = scale.clamp(1, 4);
        let (rw, rh) = (width as u32 * scale, height as u32 * scale);
        tracing::info!(
            scale,
            rw,
            rh,
            "Resolution: legacy hidpi_scale → {}x{}",
            rw,
            rh
        );
        (rw as u16, rh as u16)
    } else {
        tracing::warn!(res_mode, "Resolution: unrecognized, using logical");
        (width, height)
    };

    // Mouse coordinate mapping: RDP desktop coords → macOS logical coords
    // MouseCoordMapper maps into the picture rectangle of the capture (aspect ratio kept).
    // Only the aspect ratio of the capture matters, not its pixel size.
    // Updated by MacDisplay::request_resize() when the client negotiates a
    // different resolution.
    let coord_mapper = handler::MouseCoordMapper::new(cg_w, cg_h, width, height);
    tracing::info!(
        rdp_w = width,
        rdp_h = height,
        cg_logical_w = cg_w,
        cg_logical_h = cg_h,
        sck_w,
        sck_h,
        "Display resolution configured"
    );

    // Create input handler with coordinate mapper
    let input_handler = MacInputHandler::new(coord_mapper.clone());

    // Audio setup — shared sender slot, recreated per connection by AudioFactory
    let shared_audio_tx = if config.audio.enabled {
        Some(macrdp_audio::new_shared_audio_tx())
    } else {
        None
    };

    // Create audio factory
    let sound_factory: Option<Box<dyn ironrdp_server::SoundServerFactory>> =
        if let Some(ref shared_tx) = shared_audio_tx {
            Some(Box::new(macrdp_audio::MacAudioFactory::new(
                shared_tx.clone(),
                config.audio.sample_rate,
                config.audio.channels,
            )))
        } else {
            None
        };

    // Clipboard factory
    let cliprdr_factory: Option<Box<dyn ironrdp_server::CliprdrServerFactory>> =
        if config.clipboard.enabled {
            let temp_dir = dirs::cache_dir()
                .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
                .join("macrdp")
                .join("clipboard");
            Some(Box::new(macrdp_clipboard::MacClipboardFactory::new(
                temp_dir,
                config.clipboard.max_file_size_mb,
            )))
        } else {
            None
        };

    // fixed_resolution = true when user explicitly set resolution (not "auto")
    let fixed_resolution = (config.width > 0 && config.height > 0) || !res_auto;

    // Bitrate override: convert Mbps to bps, or None for auto-calculate
    let bitrate_override = config.bitrate_mbps.map(|mbps| mbps * 1_000_000);

    // Create shared performance stats (enabled via --perf flag)
    let perf_stats = if cli.perf {
        tracing::info!("Performance statistics collection enabled (--perf)");
        Some(perf_stats::new_shared(true))
    } else {
        None
    };

    // Create display
    let cursor_channel = config.cursor_channel.unwrap_or(true);
    let show_cursor = if cursor_channel {
        false // cursor removed from video frames when cursor channel is active
    } else {
        config.show_cursor.unwrap_or(true)
    };
    if cursor_channel {
        tracing::info!("Cursor channel enabled — cursor sent via RDP pointer PDUs");
    }
    // EGFX graphics pipeline; one per connection, shared with the display loop.
    let gfx_shared: gfx::SharedGfx = Default::default();
    // Set by the server while the client sends SuppressOutput (e.g. window minimised).
    let display_suppressed = std::sync::Arc::new(std::sync::atomic::AtomicBool::new(false));
    let display = MacDisplay::new(
        width,
        height,
        fixed_resolution,
        config.frame_rate,
        quality,
        encoder_pref,
        mode_444,
        show_cursor,
        bitrate_override,
        coord_mapper.clone(),
        shared_audio_tx.clone(),
        perf_stats.clone(),
        std::sync::Arc::clone(&gfx_shared),
        std::sync::Arc::clone(&display_suppressed),
    );

    let bind_addr: SocketAddr = format!("0.0.0.0:{}", config.port).parse()?;

    // When resolution follows the client, start each session at the size the
    // client asks for (up to the physical display size).
    let honor_client_size = (!fixed_resolution).then(|| display.max_size());

    let connection_handler: Option<Box<dyn ironrdp_server::ConnectionHandler>> =
        Some(Box::new(MacConnectionHandler {
            pending_resize: display.pending_resize(),
            coord_mapper: coord_mapper.clone(),
            audio_gate: shared_audio_tx.map(macrdp_audio::AudioPlaybackGate::new),
        }));

    let gfx_factory: Option<Box<dyn ironrdp_server::GfxServerFactory>> = Some(Box::new(
        gfx::MacGfxFactory::new(std::sync::Arc::clone(&gfx_shared)),
    ));

    // Build RDP server with Hybrid security (NLA/CredSSP)
    // This enables Windows mstsc to prompt for credentials before connecting
    let mut server = RdpServer::builder()
        .with_addr(bind_addr)
        .with_hybrid(tls_acceptor, tls_identity.pub_key)
        .with_input_handler(input_handler)
        .with_display_handler(display)
        .with_sound_factory(sound_factory)
        .with_cliprdr_factory(cliprdr_factory)
        .with_gfx_factory(gfx_factory)
        .with_connection_handler(connection_handler)
        // A new client that passes NLA takes over the running session.
        .with_connection_policy(ConnectionPolicy::Preempt)
        .with_honor_client_desktop_size(honor_client_size)
        .with_display_suppressed_handle(display_suppressed)
        .build();

    // Set credentials — required for RDP authentication
    let (username, password) = match (&config.username, &config.password) {
        (Some(u), Some(p)) => (u.clone(), p.expose().to_owned()),
        _ => {
            let random_pass = generate_password()?;
            let user = "macrdp".to_string();
            tracing::warn!("No credentials in config — using generated credentials:");
            println!("\n  ┌──────────────────────────────────┐");
            println!("  │  Username: {:<22}│", user);
            println!("  │  Password: {:<22}│", random_pass);
            println!("  └──────────────────────────────────┘\n");
            tracing::info!("Set [username] and [password] in ~/.config/macrdp/config.toml to use fixed credentials");
            (user, random_pass)
        }
    };
    server.set_credentials(Some(Credentials {
        username: username.clone(),
        password: password.clone(),
        domain: None,
    }));
    tracing::info!("Authentication configured for user: {}", username);

    tracing::info!(%bind_addr, "RDP server listening");
    tracing::info!("Connect using an RDP client (e.g., Windows mstsc or Microsoft Remote Desktop)");

    // Run server; on Ctrl-C print perf summary if enabled
    let result = tokio::select! {
        res = server.run() => res.context("RDP server error"),
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("Received Ctrl-C — shutting down");
            Ok(())
        }
    };

    if let Some(ps) = &perf_stats {
        ps.lock().unwrap().print_summary();
    }

    result
}

/// Check and request all required macOS permissions at startup.
///
/// Note: `cargo run` produces unsigned debug binaries whose path/hash changes on
/// every recompile.  macOS TCC ties permissions to the exact binary identity, so
/// the preflight check may return `false` even after the user has granted access
/// to a previous build.  We therefore only warn (never auto-open Settings) and
/// continue — actual capture or input injection will fail at runtime with a clear
/// error if the permission is truly missing.
fn check_permissions() {
    tracing::info!("Checking macOS permissions...");
    let mut all_granted = true;

    // 1. Screen Recording
    if macrdp_capture::check_screen_recording_permission() {
        tracing::info!("[OK] Screen Recording permission granted");
    } else {
        all_granted = false;
        tracing::warn!("[!!] Screen Recording — preflight check returned false");
        // Trigger the system dialog (no-op if already granted to this binary)
        macrdp_capture::request_screen_recording_permission();
        tracing::warn!(
            "     If capture fails: System Settings > Privacy & Security > Screen Recording"
        );
        tracing::warn!("     Tip: debug builds change binary identity on each compile — re-authorize after rebuild");
    }

    // 2. Accessibility (required for keyboard/mouse injection)
    if macrdp_input::check_accessibility_permission() {
        tracing::info!("[OK] Accessibility permission granted");
    } else {
        all_granted = false;
        tracing::warn!("[!!] Accessibility — preflight check returned false");
        macrdp_input::request_accessibility_permission();
        tracing::warn!("     If input fails: System Settings > Privacy & Security > Accessibility");
    }

    if all_granted {
        tracing::info!("All permissions granted");
    } else {
        tracing::warn!("Some preflight checks failed — server will start anyway.");
        tracing::warn!(
            "If you already granted permission, this is likely a debug-build identity change."
        );
    }
}

/// 16 random characters from [A-Za-z0-9], from the OS secure random source.
fn generate_password() -> Result<String> {
    const CHARSET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789";
    const LEN: usize = 16;
    let mut pass = String::with_capacity(LEN);
    let mut buf = [0u8; 32];
    while pass.len() < LEN {
        getrandom::fill(&mut buf).map_err(|e| anyhow::anyhow!("secure random source: {e}"))?;
        // Reject bytes >= 248 (4 * 62) so that each character is equally likely.
        for &b in buf.iter().filter(|&&b| b < 248) {
            if pass.len() == LEN {
                break;
            }
            pass.push(CHARSET[usize::from(b) % CHARSET.len()] as char);
        }
    }
    Ok(pass)
}

/// Hooks run once per new connection (not on a reactivation after a resize).
struct MacConnectionHandler {
    /// A resize of the previous connection does not apply to the new one.
    pending_resize: display::PendingResize,
    /// Cursor positions of the previous client do not apply to the new one.
    coord_mapper: handler::MouseCoordMapper,
    /// Stops audio capture for clients that do not want audio played on their side.
    audio_gate: Option<macrdp_audio::AudioPlaybackGate>,
}

impl ConnectionHandler for MacConnectionHandler {
    fn on_connection_info(&mut self, info: &ironrdp_server::ConnectionInfo) {
        self.pending_resize.clear();
        self.coord_mapper.forget_client_moves();
        if let Some(gate) = &mut self.audio_gate {
            gate.on_connection_info(info);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_connection_drops_old_resize_only() {
        use ironrdp_pdu::gcc::KeyboardType;
        use ironrdp_server::ConnectionInfo;

        let pending = display::PendingResize::default();
        let mapper = handler::MouseCoordMapper::new(1280, 720, 1280, 720);
        let mut handler = MacConnectionHandler {
            pending_resize: pending.clone(),
            coord_mapper: mapper.clone(),
            audio_gate: None,
        };
        let info = ConnectionInfo::new(0x409, KeyboardType::IBM_ENHANCED, String::new());

        // Left over from the previous connection: dropped when a new one starts.
        pending.set(1280, 720);
        mapper.note_client_move((640, 360));
        handler.on_connection_info(&info);
        assert_eq!(pending.take(), None);
        // The old client's cursor position is not an echo for the new client.
        assert!(!mapper.is_client_move((640, 360)));

        // Within a connection, a layout that arrives later is kept for the update loop,
        // also across the next display start (updates() does not clear it).
        pending.set(1024, 768);
        assert_eq!(pending.take(), Some((1024, 768)));
    }

    #[test]
    fn test_generate_password() {
        let a = generate_password().unwrap();
        let b = generate_password().unwrap();
        assert_eq!(a.len(), 16);
        assert!(a.bytes().all(|c| c.is_ascii_alphanumeric()), "{a}");
        assert_ne!(a, b);
    }
}
