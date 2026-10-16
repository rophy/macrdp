# macrdp

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Rust](https://img.shields.io/badge/Rust-1.94%2B-orange.svg)](https://www.rust-lang.org/)
[![macOS](https://img.shields.io/badge/macOS-14%2B-black.svg)](https://www.apple.com/macos/)

An RDP server for macOS. Connect to a Mac from Windows (mstsc), Linux (FreeRDP) or any
other RDP client that supports the RDP graphics pipeline (EGFX), preferably with H.264.

This is a fork of [x6nux/macrdp](https://github.com/x6nux/macrdp). The fork adds dynamic
resolution, session reconnect and takeover, adaptive frame rate, a separate cursor channel,
and many protocol fixes. See [Changes in this fork](#changes-in-this-fork).

---

## Why

macOS has no RDP server. Its built-in Screen Sharing (VNC) sends lossless ZRLE to
third-party viewers, so it is slow over normal networks. macrdp encodes the screen as
H.264 with the hardware encoder and sends it over RDP.

Measured on an M1 Mac (1280x720, Wi-Fi), client in a Linux VM
(method and scripts: [`tests/bench`](tests/bench/README.md)):

| | macrdp | macOS Screen Sharing |
|---|---|---|
| Full-screen motion | 47–52 fps, 12–21 Mbps | 4 fps, 44–51 Mbps |
| Scrolling | 46 fps | 16–17 fps |
| Key-to-screen latency (median) | ~50 ms | 155–205 ms |
| Static screen | ~0 Mbps | ~0 Mbps |
| Image quality | lossy (H.264 4:2:0; AVC444 available) | pixel-exact |

---

## Features

- **H.264 over RDP GFX** — AVC420, or AVC444 for full chroma. Hardware encoder
  (VideoToolbox) or software encoder (OpenH264).
- **Adaptive frame rate and bitrate** — capture drops to 5 fps when the screen is idle and
  returns to full rate on activity. Bitrate ramps up after scene changes.
- **Separate cursor channel** — the cursor is sent as RDP pointer updates, not in the video.
- **Dynamic resolution** — follows the client window size (DisplayControl channel).
- **Session reconnect and takeover** — one session at a time. A new connection that
  authenticates (NLA) replaces the current session; the old client gets the error
  "disconnected by other connection". The encoder is reused and a keyframe is replayed.
- **Audio** — system audio over `rdpsnd` (PCM 48 kHz stereo). Silent periods are not sent.
  The server respects the client's audio mode (see [Audio](#audio)).
- **Clipboard** — text and file copy/paste between client and Mac.
- **Keyboard and mouse** — scancode mapping, modifiers, wheel.
- **Lock screen** — falls back to CoreGraphics capture when ScreenCaptureKit cannot capture.
- **TLS and NLA (CredSSP)** — self-signed certificate generated on first run.
- **Clean errors** — the client gets an Error Info PDU (for example, when Screen Recording
  permission is missing) and a graceful disconnect instead of a dropped socket.

### Tested clients

| Client | Notes |
|---|---|
| Windows `mstsc` (Windows 10 22H2) | Works with default settings |
| FreeRDP 3.x with H.264 | Use `/gfx:AVC420` (or `/gfx:AVC444`); add `/sound` for audio |

The client must support the RDP graphics pipeline (EGFX). If a client opens EGFX with AVC
(H.264) disabled, for example FreeRDP `/gfx` without `:AVC420`, the server sends bitmap
updates. A client without EGFX gets no video.

---

## Requirements

- macOS 14 (Sonoma) or later. Apple Silicon is tested.
- To build from source: Rust 1.94 or later, and Xcode or the Command Line Tools.
- Permissions for the `macrdp-server` binary (System Settings → Privacy & Security):
  - **Screen Recording** — capture
  - **Accessibility** — keyboard and mouse injection

macOS grants these permissions per binary path and signature. Release binaries keep them
across upgrades; after each local rebuild, you may have to grant them again.

---

## Install

Download `macrdp-vX.Y.Z-macos-arm64.tar.gz` and `SHA256SUMS` from
[Releases](https://github.com/rophy/macrdp/releases) (Apple Silicon, signed and notarized),
then:

```bash
shasum -a 256 -c SHA256SUMS
tar xzf macrdp-vX.Y.Z-macos-arm64.tar.gz
cd macrdp-vX.Y.Z-macos-arm64
./install.sh
```

`install.sh` runs as your user (no sudo). It:

- copies `macrdp-server` to `~/.local/bin` (set `PREFIX` to change)
- creates `~/.config/macrdp/config.toml` with your user name and a random password, if it
  does not exist
- starts the LaunchAgent `io.github.rophy.macrdp`; logs go to `~/Library/Logs/macrdp/`

Then give `macrdp-server` Screen Recording and Accessibility permission and restart it
(`install.sh` prints the command). The binary is signed with a fixed identity, so the
permissions stay valid after an upgrade. `./uninstall.sh` removes the binary and the
LaunchAgent and keeps your config and logs.

---

## Build and run

```bash
cargo build --release --bin macrdp-server
./target/release/macrdp-server --config config.toml
```

Connect your RDP client to `<mac-ip>:<port>`.

Command-line options:

| Option | Meaning |
|---|---|
| `-c, --config <path>` | Config file |
| `-p, --port <port>` | Listen port (default 3389; overrides the config file) |
| `--frame-rate <n>` | Target frame rate |
| `--log-level <level>` | `trace`, `debug`, `info`, `warn`, `error` |
| `--perf` | Collect performance statistics and print a summary on exit |

---

## Configuration

Copy [`config.example.toml`](config.example.toml) to `config.toml` and edit it.

macrdp uses the first `config.toml` it finds in:

1. the current directory
2. `~/.config/macrdp/`
3. `~/Library/Application Support/macrdp/`

| Key | Default | Meaning |
|---|---|---|
| `port` | `3389` | TCP port |
| `username`, `password` | none | Login. If not set, the server makes a random password for user `macrdp` at each start and prints it to stdout. |
| `width`, `height` | `0` (auto) | Fixed resolution. `0` follows the display and the client. |
| `resolution` | `auto` | Capture size, e.g. `"3840x2160"` for a sharper image on Retina displays |
| `frame_rate` | `60` | Target frame rate |
| `quality` | `high_quality` | `low_latency`, `balanced` or `high_quality` |
| `encoder` | software | `hardware` (VideoToolbox) or `software` (OpenH264) |
| `chroma_mode` | `avc420` | `avc420`, or `avc444` for full chroma |
| `bitrate_mbps` | auto | Bitrate ceiling |
| `cursor_channel` | `true` | Send the cursor as pointer updates instead of in the video |
| `show_cursor` | auto | Draw the cursor in the video (default: only when `cursor_channel` is off) |
| `cert_path`, `key_path` | auto | TLS certificate and key (PEM); generated if not set |
| `idle_timeout_secs` | `1800` | Disconnect after this many seconds without input |
| `log_level` | `info` | Log level |
| `[audio] enabled` | `true` | Audio forwarding |
| `[audio] sample_rate`, `channels` | `48000`, `2` | Audio format |
| `[clipboard] enabled` | `true` | Clipboard sync |
| `[clipboard] max_file_size_mb` | `100` | Largest file for clipboard file transfer |

---

## Audio

The server sends audio only when the client asks for it. It reads the audio mode from the
client's Client Info PDU:

| Client setting | mstsc "Remote audio" | FreeRDP | Server |
|---|---|---|---|
| Play on this computer | default | `/sound` | sends audio |
| Play on remote computer | "Play on remote computer" | `/audio-mode:1` | no audio; the Mac plays it |
| Do not play | "Do not play" | default, `/audio-mode:2` | no audio |

When no audio is requested, the Mac does not capture audio for that session.

Audio is uncompressed PCM, about 1.5 Mbps while sound plays. Silence is not sent.

---

## Project layout

```
crates/
├── macrdp-server/                  server binary: config, session, display loop
├── macrdp-capture/                 ScreenCaptureKit capture (+ CoreGraphics fallback)
├── macrdp-encode/                  H.264 encoders (VideoToolbox, OpenH264), AVC444
├── macrdp-input/                   keyboard and mouse injection
├── macrdp-audio/                   system audio → rdpsnd
└── macrdp-clipboard/               clipboard and file transfer
tests/
├── e2e_test.sh                     smoke test from a Linux client
└── bench/                          macrdp vs macOS Screen Sharing benchmark
```

---

## Tests

```bash
cargo test --workspace
```

`MACRDP_USER=… MACRDP_PASS=… tests/e2e_test.sh <host> [port]` runs a smoke test with
FreeRDP 3 (with H.264) from a Linux machine: login, session takeover, bitmap, AVC420,
no-audio and dynamic-resolution sessions. `XFREERDP` selects the client binary. Optional:
`MACRDP_AVC444=1` (server runs with `chroma_mode = "avc444"`) adds an AVC444 test,
`MACRDP_LOG` (server log readable by the script) checks the no-audio log line, and
`MACRDP_MAC_SHELL` (a command that runs a script on the Mac, e.g. `ssh user@mac sh`) adds a
clipboard file copy test. The script header has the details.
`tests/bench` has the benchmark (see its README).

---

## Changes in this fork

Compared with upstream:

- Built on IronRDP upstream `master`: H.264 goes through its EGFX server (`ironrdp-egfx`,
  with ZGFX compression and frame-acknowledge flow control)
- Dynamic resolution through the DisplayControl channel, with a new EGFX surface at each
  size
- Session reconnect (warm reset, encoder reuse, keyframe replay) and session takeover
  (a new authenticated connection replaces the current one)
- Activity-based adaptive frame rate; fix: capture size is kept when the frame rate changes
- Adaptive bitrate (from the frame-acknowledge round trip) and progressive rendering after
  scene changes
- Encoding stops while the client does not show the output (Suppress Output PDU)
- Separate cursor channel (RDP pointer PDUs)
- Bitmap updates for EGFX clients with AVC disabled
- Audio: no silent chunks; respects the client audio mode
- Error Info PDU when screen capture cannot start
- Main display is used for capture (was: first display)
- Signed and notarized Apple Silicon releases, install script and LaunchAgent; CI
- Desktop UI (Tauri) removed; the server runs as a LaunchAgent
- E2E smoke test and benchmark scripts

---

## Versioning and releases

The version is in `Cargo.toml` (`[workspace.package] version`). Between releases, `main`
carries the next version with `-dev` (for example `0.2.0-dev`). `macrdp-server --version`
adds `git describe` output to builds that are not a release tag.

To release: commit `version = "X.Y.Z"` to `main`. After CI passes, the release workflow
builds, signs, notarizes and publishes `vX.Y.Z` and creates the tag. Then commit the next
`-dev` version. A `-dev` version, or a version whose tag exists, is not released.

---

## Credits

- **[x6nux/macrdp](https://github.com/x6nux/macrdp)** — the upstream project.
- **[IronRDP](https://github.com/Devolutions/IronRDP)** — Rust RDP implementation. macrdp
  uses the `macrdp-v2` branch of the fork [rophy/IronRDP](https://github.com/rophy/IronRDP):
  upstream `master` plus three small changes to the server and acceptor crates (logon
  failure status for a wrong NLA password, no rdpsnd when the client asks for no audio,
  Error Info when the display cannot start).
- **[FreeRDP](https://github.com/FreeRDP/FreeRDP)** — reference for AVC444 encoding and
  the client used for testing.
- **[RustDesk](https://github.com/rustdesk/rustdesk)** — ideas for capture and input.

---

## License

GNU General Public License v3.0. See [LICENSE](LICENSE).
