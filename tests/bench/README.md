# macrdp vs macOS Screen Sharing benchmark

Compares macrdp (RDP, H.264) with the built-in Screen Sharing (VNC) from a Linux client VM.

## What it measures

The Mac shows `bench.html` in a full-screen Chrome kiosk. Pages (`#mode`):

- `static` – idle bandwidth + image quality (PSNR/SSIM vs the Mac's own screenshot)
- `motion` – full-screen animation; a Gray-code frame counter gives shown fps
- `scroll` – scrolling text document
- `latency` / `latencysmall` – a key press flips a 200x200 / 12x18 patch; key-to-screen time

Server CPU comes from `ps` CPU-time deltas on the Mac (`mac/cpu.sh`).

## Parts

| File | Runs on | Job |
|---|---|---|
| `orch.py` | host | runs the matrix, writes `results.json` + screenshots |
| `bclient.py` | client VM (Xvfb `:99`, 1280x720) | samples the screen, sends keys |
| `sess.sh` | client VM | starts FreeRDP or TigerVNC full screen |
| `ardproxy.py` | client VM | does Apple auth (type 30) upstream, offers "None" to TigerVNC |
| `rfbparse.py` | client VM | counts encodings in the VNC stream dump (`/tmp/s2c_*.bin`) |
| `quality.py` | anywhere with numpy | PSNR/SSIM vs reference |
| `mac/*.sh` | Mac | open page, screenshot via launchd, CPU probe |

## Run

Needs: a multipass VM with Xvfb, openbox, FreeRDP with H.264, `tigervnc-viewer`,
`python3-xlib`, `python3-cryptography`; Chrome on the Mac; Screen Sharing enabled.
`mac/shot.sh` needs a Screen Recording TCC grant for `/usr/sbin/screencapture`.

```sh
MAC_SSH=user@mac MAC_HOST=<mac-ip> RDP_PORT=3389 RDP_USER=... RDP_PASS=... \
VNC_USER=<macos-user> VNC_PASS=<macos-password> \
python3 tests/bench/orch.py /tmp/bench-out 2
```

Optional: `VM`, `BENCH_CONFIGS` (default `rdp vnc vncjpeg`), `FPS_SECS`, `LAT_TRIALS`.

Note: macOS Screen Sharing answers third-party viewers with ZRLE only, even when the
viewer asks for Tight/JPEG, so `vnc` and `vncjpeg` give the same results.
