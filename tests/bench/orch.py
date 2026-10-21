#!/usr/bin/env python3
"""Run the macrdp benchmark from the host: a FreeRDP client on a multipass VM, bench.html on the Mac.
usage: orch.py <out-dir> [rounds]   (see README.md for the env vars)"""
import json, os, re, subprocess, sys, time

MAC = os.environ["MAC_SSH"]          # ssh target of the Mac, e.g. user@host
VM = os.environ.get("VM", "ubuntu-rdp")
XFER = os.path.expanduser(os.environ.get("XFER_DIR", "~/bench-xfer"))  # multipass (snap) cannot read /tmp
MAC_DIR = os.environ.get("MAC_DIR", "/tmp/macrdp-bench")  # bench page, scripts, Chrome profile on the Mac
DISPLAY = os.environ.get("BENCH_DISPLAY", ":99")          # Xvfb display on the VM (1280x720)
RDP_PORT = os.environ.get("RDP_PORT", "3390")
STATS_PORT = os.environ.get("STATS_PORT", "")            # server's --stats-endpoint port, if on
# No password here: RDP_PASS_FILE is a file on the VM, read by sess.sh.
SESS_ENV = " ".join(f'{k}="{os.environ.get(k, "")}"'
                    for k in ("MAC_HOST", "RDP_PORT", "RDP_USER", "RDP_PASS_FILE", "BENCH_DISPLAY"))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
MODES = os.environ.get("BENCH_MODES", "static motion scroll latency latencysmall").split()
QUALITY = os.environ.get("BENCH_QUALITY") == "1"
FPS_SECS = int(os.environ.get("FPS_SECS", 20))
LAT_TRIALS = int(os.environ.get("LAT_TRIALS", 30))
os.makedirs(OUT, exist_ok=True)


def sh(cmd, timeout=600):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout).stdout


def mac(cmd, timeout=120):
    return sh(f"ssh {MAC} '{cmd}'", timeout)


def vm(cmd, timeout=600):
    return sh(f"multipass exec {VM} -- bash -c '{cmd}'", timeout)


def deploy():
    os.makedirs(XFER, exist_ok=True)
    for f in ("bclient.py", "sess.sh"):
        sh(f"cp {HERE}/{f} {XFER}/{f} && multipass transfer {XFER}/{f} {VM}:/tmp/macrdp-bench-{f}")
    mac(f"mkdir -p {MAC_DIR}")
    sh(f"scp -q {HERE}/bench.html {HERE}/mac/page.sh {HERE}/mac/shot.sh {HERE}/mac/cpu.sh {MAC}:{MAC_DIR}/")
    # Own Xvfb display (with a window manager for the full-screen client) unless it runs already.
    vm(f"chmod +x /tmp/macrdp-bench-sess.sh; if ! pgrep -f \"^Xvfb {DISPLAY} \" >/dev/null; then"
       f" (setsid nohup Xvfb {DISPLAY} -screen 0 1280x720x24 >/dev/null 2>&1 &); sleep 1;"
       f" (DISPLAY={DISPLAY} setsid nohup openbox >/dev/null 2>&1 &); fi")


def cpu_snapshot():
    snap = {}
    for line in mac(f"{MAC_DIR}/cpu.sh {RDP_PORT}").splitlines():
        pid, secs, name = line.split(" ", 2)
        snap[pid] = (float(secs), name.rsplit("/", 1)[-1])
    return snap


def cpu_delta(a, b, wall):
    """CPU% (of one core) per process group over the window."""
    groups = {}
    for pid, (secs, name) in b.items():
        prev = a[pid][0] if pid in a else 0.0
        groups[name] = groups.get(name, 0.0) + (secs - prev)
    return {k: round(v / wall * 100, 1) for k, v in groups.items()}


def server_stats():
    """One snapshot of the server's --stats-endpoint (None if off)."""
    if not STATS_PORT:
        return None
    try:
        return json.loads(mac(f"nc 127.0.0.1 {STATS_PORT}", 15).strip().splitlines()[-1])
    except (ValueError, IndexError):
        return None


def stats_delta(a, b, wall):
    if not a or not b:
        return None
    # frames_sent moves only with --adaptive-bitrate or for a client that sends EGFX QoE
    # reports (mstsc; FreeRDP does not): None when it never moved.
    fps = round((b["frames_sent"] - a["frames_sent"]) / wall, 2) if b["frames_sent"] else None
    return {"server_fps": fps,
            "bitrate_mbps": round(b["bitrate_bps"] / 1e6, 2), "ceiling_mbps": round(b["ceiling_bps"] / 1e6, 2),
            "rtt_ms": b["rtt_ms"], "queue_delay_ms": b["queue_delay_ms"], "fps_cap": b["fps"],
            "size": f'{b["width"]}x{b["height"]}'}


def run_workload(mode, rnd):
    mac(f"{MAC_DIR}/page.sh {mode}")
    time.sleep(4)  # let page + adaptive logic settle
    kind = "latency" if mode.startswith("latency") else "fps"
    arg = LAT_TRIALS if kind == "latency" else FPS_SECS
    a, sa, ta = cpu_snapshot(), server_stats(), time.time()
    top_secs = FPS_SECS if kind == "fps" else 15
    top = subprocess.Popen(f"ssh {MAC} 'top -l 2 -s {top_secs} -n 0 | grep \"CPU usage\" | tail -1'",
                           shell=True, stdout=subprocess.PIPE, text=True)
    res = json.loads(vm(f"DISPLAY={DISPLAY} python3 /tmp/macrdp-bench-bclient.py {kind} {arg} /tmp/macrdp-bench-r.json")
                     .strip().splitlines()[-1])
    b, sb, tb = cpu_snapshot(), server_stats(), time.time()
    res["server_cpu_pct"] = cpu_delta(a, b, tb - ta)
    res["server_stats"] = stats_delta(sa, sb, tb - ta)
    m = re.search(r"([\d.]+)% idle", top.communicate()[0] or "")
    res["mac_total_busy_pct"] = round(100 - float(m.group(1)), 1) if m else None
    res.update(config="rdp", workload=mode, round=rnd)
    print(json.dumps(res), flush=True)
    return res


def quality_shot(rnd):
    mac(f"{MAC_DIR}/page.sh static")
    time.sleep(5)
    vm(f"DISPLAY={DISPLAY} python3 /tmp/macrdp-bench-bclient.py shot /tmp/macrdp-bench-q.png")
    dst = f"{OUT}/q_rdp_{rnd}.png"
    sh(f"multipass transfer {VM}:/tmp/macrdp-bench-q.png {XFER}/q.png && cp {XFER}/q.png {dst}")
    if rnd == 0:
        mac(f"{MAC_DIR}/shot.sh {MAC_DIR}/ref_static.png")
        sh(f"scp -q {MAC}:{MAC_DIR}/ref_static.png {OUT}/ref_static.png")


deploy()
results = []
for rnd in range(ROUNDS):
    vm(f"{SESS_ENV} /tmp/macrdp-bench-sess.sh rdp")
    time.sleep(8)
    if QUALITY:
        quality_shot(rnd)
    for mode in MODES:
        results.append(run_workload(mode, rnd))
        json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
vm(f"BENCH_DISPLAY={DISPLAY} /tmp/macrdp-bench-sess.sh stop")
mac(f"{MAC_DIR}/page.sh close")
print("DONE")
