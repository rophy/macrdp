use clap::Parser;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "macrdp", about = "macOS RDP Server")]
pub struct Cli {
    /// TCP port to listen on
    #[arg(short, long, default_value_t = 3389)]
    pub port: u16,

    /// Path to config file
    #[arg(short, long)]
    pub config: Option<PathBuf>,

    /// Target frame rate (30, 60, or 120)
    #[arg(long)]
    pub frame_rate: Option<u32>,

    /// Log level: trace, debug, info, warn, error
    #[arg(long)]
    pub log_level: Option<String>,

    /// Enable performance statistics collection and print summary on exit
    #[arg(long, default_value_t = false)]
    pub perf: bool,
}

/// Audio forwarding configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct AudioConfig {
    pub enabled: bool,
    pub sample_rate: u32,
    pub channels: u16,
}

impl Default for AudioConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sample_rate: 48000,
            channels: 2,
        }
    }
}

/// Clipboard synchronization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct ClipboardConfig {
    pub enabled: bool,
    pub file_transfer: bool,
    pub max_file_size_mb: u32,
}

impl Default for ClipboardConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            file_transfer: true,
            max_file_size_mb: 100,
        }
    }
}

#[derive(Debug, Deserialize)]
#[serde(default)]
pub struct ServerConfig {
    pub port: u16,
    pub frame_rate: u32,
    /// Resolution width (0 = auto-detect from display)
    pub width: u32,
    /// Resolution height (0 = auto-detect from display)
    pub height: u32,
    pub username: Option<String>,
    pub password: Option<String>,
    pub cert_path: Option<PathBuf>,
    pub key_path: Option<PathBuf>,
    pub idle_timeout_secs: u64,
    /// Log level: trace, debug, info, warn, error
    pub log_level: Option<String>,
    /// Video quality: low_latency, balanced, high_quality (default: high_quality)
    pub quality: Option<String>,
    /// H.264 encoder: software, hardware, auto (default: software)
    /// - software: OpenH264 CPU encoder (P-frame support, ~40ms latency)
    /// - hardware: VideoToolbox GPU encoder (IDR-only, ~6ms latency, higher bandwidth)
    /// - auto: same as software
    pub encoder: Option<String>,
    /// Chroma subsampling mode: "avc420" or "avc444" (default: "avc420")
    /// - avc420: standard 4:2:0 chroma (compatible with all RDP clients)
    /// - avc444: full 4:4:4 chroma via dual-stream AVC444 (requires V10+ client, best quality)
    pub chroma_mode: Option<String>,
    /// Resolution: "auto" or "WxH" like "3840x2160" (default: "auto")
    #[serde(alias = "hidpi_scale")]
    pub resolution: Option<String>,
    /// Show cursor in capture (default: true when cursor_channel is off)
    pub show_cursor: Option<bool>,
    /// Send cursor position via separate RDP pointer PDUs (default: true)
    /// When enabled, cursor is removed from video frames and sent as lightweight
    /// PointerPosition updates at ~60Hz, giving zero-latency cursor movement.
    pub cursor_channel: Option<bool>,
    /// Target bitrate in Mbps (default: auto-calculated from resolution/fps/quality)
    /// Override this to force a specific bitrate, e.g. 50 for 50 Mbps.
    pub bitrate_mbps: Option<u32>,
    /// Skip encoding when screen content is unchanged (default: true)
    /// When enabled, idle frames from ScreenCaptureKit are not encoded,
    /// reducing CPU/GPU usage on static screens.
    pub skip_unchanged: Option<bool>,
    /// Seconds between keepalive IDR frames during screen idle (default: 2)
    /// Prevents RDP client timeout when no frames are being sent.
    pub idle_keyframe_sec: Option<u32>,
    /// Audio forwarding configuration
    #[serde(default)]
    pub audio: AudioConfig,
    /// Clipboard synchronization configuration
    #[serde(default)]
    pub clipboard: ClipboardConfig,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            port: 3389,
            frame_rate: 60,
            width: 0,
            height: 0,
            username: None,
            password: None,
            cert_path: None,
            key_path: None,
            idle_timeout_secs: 1800,
            log_level: None,
            quality: None,
            encoder: None,
            chroma_mode: None,
            resolution: None,
            show_cursor: None,
            cursor_channel: None,
            bitrate_mbps: None,
            skip_unchanged: None,
            idle_keyframe_sec: None,
            audio: AudioConfig::default(),
            clipboard: ClipboardConfig::default(),
        }
    }
}

impl ServerConfig {
    /// Load config from file, then apply CLI overrides
    pub fn load(cli: &Cli) -> anyhow::Result<Self> {
        let mut config = if let Some(path) = &cli.config {
            tracing::info!(?path, "Loading config from CLI-specified path");
            let content = std::fs::read_to_string(path)?;
            toml::from_str(&content)?
        } else {
            let default_path = config_dir().join("config.toml");
            if default_path.exists() {
                tracing::info!(path = %default_path.display(), "Loading config");
                let content = std::fs::read_to_string(&default_path)?;
                toml::from_str(&content)?
            } else {
                tracing::info!(
                    search_paths = ?config_dir_candidates().iter().map(|p| p.join("config.toml")).collect::<Vec<_>>(),
                    "No config.toml found, using defaults"
                );
                ServerConfig::default()
            }
        };

        // CLI overrides
        if cli.port != 3389 {
            config.port = cli.port;
        }
        if let Some(fps) = cli.frame_rate {
            config.frame_rate = fps;
        }
        if let Some(level) = &cli.log_level {
            config.log_level = Some(level.clone());
        }

        Ok(config)
    }
}

/// Returns the macrdp config directory.
/// Search order (first existing directory wins):
///   1. ./  (current working directory)
///   2. ~/.config/macrdp  (XDG convention)
///   3. ~/Library/Application Support/macrdp  (macOS native)
///
/// For writes (TLS cert generation etc.), uses the first writable candidate.
pub fn config_dir() -> PathBuf {
    let candidates = config_dir_candidates();
    for dir in &candidates {
        if dir.join("config.toml").exists() {
            return dir.clone();
        }
    }
    // No config found — return the first candidate for writes
    candidates.into_iter().next().unwrap_or_else(|| PathBuf::from("."))
}

/// All candidate config directories in priority order.
fn config_dir_candidates() -> Vec<PathBuf> {
    let mut dirs_list = Vec::new();

    // 1. Current working directory
    if let Ok(cwd) = std::env::current_dir() {
        dirs_list.push(cwd);
    } else {
        dirs_list.push(PathBuf::from("."));
    }

    // 2. ~/.config/macrdp (XDG)
    if let Some(home) = dirs::home_dir() {
        dirs_list.push(home.join(".config").join("macrdp"));
    }

    // 3. macOS native (~/Library/Application Support/macrdp)
    if let Some(native) = dirs::config_dir() {
        dirs_list.push(native.join("macrdp"));
    }

    dirs_list
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_default_config() {
        let config = ServerConfig::default();
        assert_eq!(config.port, 3389);
        assert_eq!(config.frame_rate, 60);
        assert!(config.username.is_none());
    }

    #[test]
    fn test_parse_encoding_config() {
        let toml_str = r#"
            port = 3389
            skip_unchanged = false
            idle_keyframe_sec = 5
        "#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.skip_unchanged, Some(false));
        assert_eq!(config.idle_keyframe_sec, Some(5));
    }

    #[test]
    fn test_default_encoding_config() {
        let config = ServerConfig::default();
        assert!(config.skip_unchanged.is_none());
        assert!(config.idle_keyframe_sec.is_none());
    }

    #[test]
    fn test_parse_toml_config() {
        let toml_str = r#"
            port = 13389
            frame_rate = 120
            username = "admin"
            password = "secret"
        "#;
        let config: ServerConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.port, 13389);
        assert_eq!(config.frame_rate, 120);
        assert_eq!(config.username.as_deref(), Some("admin"));
        assert_eq!(config.password.as_deref(), Some("secret"));
    }
}
