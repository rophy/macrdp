# macrdp benchmark

Measures what a user of a FreeRDP client sees: shown frame rate, frame gaps, bandwidth and
key-to-screen latency, plus server CPU and the server's own stats.

## What it measures

The Mac shows `bench.html` in a full-screen Chrome kiosk. Pages (`#mode`):

- `static` – idle bandwidth (and, with `BENCH_QUALITY=1`, image quality: PSNR/SSIM vs the Mac's
  own screenshot, `quality.py`)
- `motion` – full-screen animation; a Gray-code frame counter gives shown fps
- `scroll` – scrolling text document
- `latency` / `latencysmall` – a key press flips a 200x200 / 12x18 patch; key-to-screen time

The Gray-code counter wraps after 4096 frames (about 68 s at 60 fps): keep `FPS_SECS` below that.

Per workload, `results.json` gets:

| Field | From |
|---|---|
| `shown_fps`, `source_fps`, `delivered_ratio`, `gap_ms`, `big_gaps_over_100ms` | client screen (`bclient.py fps`) |
| `latency_ms` (`p50`, `p90`, ...), `timeouts` | client screen (`bclient.py latency`) |
| `rx_mbps`, `tx_mbps` | client NIC counters |
| `server_cpu_pct` | `ps` CPU time deltas on the Mac (`mac/cpu.sh`: the process on `RDP_PORT`, WindowServer, replayd) |
| `mac_total_busy_pct` | `top` on the Mac |
| `server_stats` | the server's `--stats-endpoint` (H.264 frames sent per second, bitrate, ceiling, RTT, queue delay), when `STATS_PORT` is set |

## Parts

| File | Runs on | Job |
|---|---|---|
| `orch.py` | host | runs the workloads, writes `results.json` (+ screenshots with `BENCH_QUALITY=1`) |
| `bclient.py` | client VM (Xvfb, 1280x720) | samples the screen, sends keys |
| `sess.sh` | client VM | starts FreeRDP full screen (AVC420) |
| `quality.py` | anywhere with numpy | PSNR/SSIM vs reference |
| `mac/*.sh` | Mac | open/close the page, CPU probe, screenshot via launchd |

## Run

Needs: a multipass VM with Xvfb, openbox, FreeRDP 3 with H.264 at `/usr/local/bin/xfreerdp`,
`python3-xlib` (and `python3-pil` for screenshots); Chrome on the Mac. The Mac display and the
client display should both be 1280x720 (the client reads the page 1:1). `mac/shot.sh` needs a
Screen Recording TCC grant for `/usr/sbin/screencapture`.

Start the server with H.264 and, for `server_stats`, the stats endpoint:

```sh
macrdp --stats-endpoint --bind 0.0.0.0:3390 --password-file ~/.macrdp-pw   # or --keychain
```

Put the account password into a file on the VM (mode 0600; it goes to FreeRDP on stdin, never
on a command line), then on the host:

```sh
MAC_SSH=user@mac MAC_HOST=<mac-ip> RDP_PORT=3390 RDP_USER=<mac user> \
RDP_PASS_FILE=/home/ubuntu/macrdp-pw STATS_PORT=40245 \
python3 tests/bench/orch.py /tmp/bench-out 1
```

Optional: `VM` (default `ubuntu-rdp`), `BENCH_DISPLAY` (Xvfb display on the VM, default `:99`),
`MAC_DIR` (folder on the Mac for the page, scripts and the Chrome profile, default
`/tmp/macrdp-bench`), `BENCH_MODES`, `FPS_SECS` (default 20), `LAT_TRIALS` (default 30),
`BENCH_QUALITY=1`. `STATS_PORT` is the server's `MACRDP_STATS_PORT` (default 40245).

Notes:

- The Mac Wi-Fi radio sleeps on an idle link and adds 50-200 ms to the first packet after a pause.
  The server's Wi-Fi keep-awake (default on) hides most of it; compare with
  `--no-wifi-keep-awake` when latency numbers matter.
- `server_stats` covers the H.264 path only (frames sent through EGFX). The server updates its
  frame counter with adaptive bitrate (the default) or for a client that sends EGFX QoE reports
  (mstsc; FreeRDP does not), so with FreeRDP and `--no-adaptive-bitrate` `server_fps` is `null`.
- The macOS Screen Sharing comparison of macrdp-v1 (`ardproxy.py`, `rfbparse.py`, the `vnc`
  configs) is not part of this port.
