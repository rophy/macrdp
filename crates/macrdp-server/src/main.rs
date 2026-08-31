mod config;
mod display;
mod handler;
mod perf_stats;
mod tls;

use anyhow::{Context, Result};
use clap::Parser;
use config::{Cli, ServerConfig};
use display::MacDisplay;
use handler::MacInputHandler;
use ironrdp_server::{Credentials, RdpServer, TlsIdentityCtx, gfx::GfxState};
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = ServerConfig::load(&cli)?;

    // Initialize logging — write to both stderr and file
    let log_level = config.log_level.as_deref().unwrap_or("info");
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new(log_level));

    let log_path = std::env::current_dir().unwrap_or_default().join("macrdp.log");
    let log_file = std::fs::File::create(&log_path)?;

    use tracing_subscriber::layer::SubscriberExt;
    use tracing_subscriber::util::SubscriberInitExt;

    tracing_subscriber::registry()
        .with(env_filter)
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
        )
        .with(
            tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_writer(std::sync::Mutex::new(log_file))
        )
        .init();

    eprintln!("Log file: {}", log_path.display());

    tracing::info!(?config, "macrdp server starting");

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
            tracing::info!(width = w, height = h, "CG logical display size (CGEvent coordinate space)");
            (w as u16, h as u16)
        }
        Err(e) => {
            tracing::warn!("Failed to detect CG display size: {e}, falling back to SCK size");
            (sck_w, sck_h)
        }
    };
    if sck_w != cg_w || sck_h != cg_h {
        tracing::warn!(
            sck_w, sck_h, cg_w, cg_h,
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
    tracing::info!(chroma_mode = config.chroma_mode.as_deref().unwrap_or("avc420"), mode_444, "Chroma mode");

    // Resolution: "auto", "WxH" (e.g. "3840x2160"), or legacy scale (e.g. "2")
    let res_mode = config.resolution.as_deref().unwrap_or("auto");
    let res_auto = res_mode == "auto";
    let (width, height) = if res_mode == "auto" {
        let scale = macrdp_capture::detect_display_scale().unwrap_or(1);
        tracing::info!(scale, width, height, "Resolution auto: display scale");
        (width * scale as u16, height * scale as u16)
    } else if let Some((w, h)) = res_mode.split_once('x').and_then(|(w, h)| {
        Some((w.parse::<u16>().ok()?, h.parse::<u16>().ok()?))
    }) {
        tracing::info!(w, h, "Resolution: explicit WxH");
        (w, h)
    } else if let Ok(scale) = res_mode.parse::<u32>() {
        let scale = scale.max(1).min(4);
        let (rw, rh) = (width as u32 * scale, height as u32 * scale);
        tracing::info!(scale, rw, rh, "Resolution: legacy hidpi_scale → {}x{}", rw, rh);
        (rw as u16, rh as u16)
    } else {
        tracing::warn!(res_mode, "Resolution: unrecognized, using logical");
        (width, height)
    };

    // Mouse coordinate mapping: RDP desktop coords → macOS logical coords
    // MouseCoordMapper maps proportionally: mac = rdp × logical ÷ rdp_desktop
    // Completely independent of capture/encode resolution.
    // Updated by MacDisplay::request_resize() when the client negotiates a
    // different resolution.
    let coord_mapper = handler::MouseCoordMapper::new(cg_w, cg_h, width, height);
    tracing::info!(
        rdp_w = width, rdp_h = height,
        cg_logical_w = cg_w, cg_logical_h = cg_h,
        sck_w, sck_h,
        "Display resolution configured"
    );

    // Create shared GFX state
    let gfx_state = Arc::new(Mutex::new(GfxState::new(width, height, mode_444)));

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
            Some(Box::new(macrdp_clipboard::MacClipboardFactory::new(temp_dir, config.clipboard.max_file_size_mb)))
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

    // Create display with shared GFX state
    let cursor_channel = config.cursor_channel.unwrap_or(true);
    let show_cursor = if cursor_channel {
        false // cursor removed from video frames when cursor channel is active
    } else {
        config.show_cursor.unwrap_or(true)
    };
    if cursor_channel {
        tracing::info!("Cursor channel enabled — cursor sent via RDP pointer PDUs");
    }
    let display = MacDisplay::new(width, height, fixed_resolution, config.frame_rate, quality, encoder_pref, mode_444, show_cursor, bitrate_override, Arc::clone(&gfx_state), coord_mapper, shared_audio_tx, perf_stats.clone());

    let bind_addr: SocketAddr = format!("0.0.0.0:{}", config.port).parse()?;

    // Build RDP server with Hybrid security (NLA/CredSSP)
    // This enables Windows mstsc to prompt for credentials before connecting
    let mut server = RdpServer::builder()
        .with_addr(bind_addr)
        .with_hybrid(tls_acceptor, tls_identity.pub_key)
        .with_input_handler(input_handler)
        .with_display_handler(display)
        .with_sound_factory(sound_factory)
        .with_cliprdr_factory(cliprdr_factory)
        .build();

    // Share GFX state with the server
    server.set_gfx_state(gfx_state);

    // Multi-monitor support
    if let Some(max) = config.max_monitors {
        server.set_max_monitors(max);
        tracing::info!(max_monitors = max, "Multi-monitor support configured");
    }

    // Set credentials — required for RDP authentication
    let (username, password) = match (&config.username, &config.password) {
        (Some(u), Some(p)) => (u.clone(), p.clone()),
        _ => {
            // Generate a random password from PID + timestamp entropy
            let seed = std::process::id() as u64
                ^ std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap()
                    .as_nanos() as u64;
            let random_pass: String = (0..8)
                .map(|i| {
                    let v = ((seed.wrapping_mul(6364136223846793005).wrapping_add(i * 1442695040888963407)) >> (i * 5 + 3)) % 36;
                    if v < 10 { (b'0' + v as u8) as char } else { (b'a' + (v - 10) as u8) as char }
                })
                .collect();
            let user = "macrdp".to_string();
            tracing::warn!("No credentials in config — using generated credentials:");
            println!("\n  ┌──────────────────────────────────┐");
            println!("  │  Username: {:<22}│", &user);
            println!("  │  Password: {:<22}│", &random_pass);
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
        tracing::warn!("     If capture fails: System Settings > Privacy & Security > Screen Recording");
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
        tracing::warn!("If you already granted permission, this is likely a debug-build identity change.");
    }
}
