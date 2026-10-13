#!/usr/bin/env python3
"""Run the macrdp vs macOS Screen Sharing benchmark matrix from the host.
usage: orch.py <out-dir> [rounds]   (see README.md for the env vars)"""
import json, os, re, subprocess, sys, time

MAC = os.environ["MAC_SSH"]          # ssh target of the Mac, e.g. user@host
VM = os.environ.get("VM", "ubuntu-rdp")
XFER = os.path.expanduser(os.environ.get("XFER_DIR", "~/bench-xfer"))  # multipass (snap) cannot read /tmp
SESS_ENV = " ".join(f'{k}="{os.environ.get(k, "")}"'
                    for k in ("MAC_HOST", "RDP_PORT", "RDP_USER", "RDP_PASS", "VNC_USER", "VNC_PASS"))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1]
ROUNDS = int(sys.argv[2]) if len(sys.argv) > 2 else 2
CONFIGS = os.environ.get("BENCH_CONFIGS", "rdp vnc vncjpeg").split()
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
    for f in ("bclient.py", "ardproxy.py", "sess.sh"):
        sh(f"cp {HERE}/{f} {XFER}/{f} && multipass transfer {XFER}/{f} {VM}:/tmp/{f}")
    sh(f"scp -q {HERE}/bench.html {HERE}/mac/page.sh {HERE}/mac/shot.sh {HERE}/mac/cpu.sh {MAC}:/tmp/")
    vm("chmod +x /tmp/sess.sh; pgrep -x Xvfb >/dev/null || (setsid nohup Xvfb :99 -screen 0 1280x720x24 >/dev/null 2>&1 &);"
       " sleep 1; pgrep -x openbox >/dev/null || (DISPLAY=:99 setsid nohup openbox >/dev/null 2>&1 &)")


def cpu_snapshot():
    snap = {}
    for line in mac("/tmp/cpu.sh").splitlines():
        pid, secs, name = line.split(" ", 2)
        snap[pid] = (float(secs), name.rsplit("/", 1)[-1])
    return snap


def cpu_delta(a, b, wall):
    """CPU% (of one core) per process group over the window."""
    groups = {}
    for pid, (secs, name) in b.items():
        if "debug/macrdp" in name:
            continue
        prev = a.get(pid, (0.0, name))[0] if pid in a else 0.0
        key = "macrdp-server" if "macrdp-server" in name else name
        groups[key] = groups.get(key, 0.0) + (secs - prev)
    return {k: round(v / wall * 100, 1) for k, v in groups.items()}


def run_workload(cfg, mode, rnd):
    mac(f"/tmp/page.sh {mode}")
    time.sleep(4)  # let page + adaptive logic settle
    kind = "latency" if mode.startswith("latency") else "fps"
    arg = LAT_TRIALS if kind == "latency" else FPS_SECS
    a, ta = cpu_snapshot(), time.time()
    top_secs = FPS_SECS if kind == "fps" else 15
    top = subprocess.Popen(f"ssh {MAC} 'top -l 2 -s {top_secs} -n 0 | grep \"CPU usage\" | tail -1'",
                           shell=True, stdout=subprocess.PIPE, text=True)
    res = json.loads(vm(f"DISPLAY=:99 python3 /tmp/bclient.py {kind} {arg} /tmp/r.json").strip().splitlines()[-1])
    b, tb = cpu_snapshot(), time.time()
    res["server_cpu_pct"] = cpu_delta(a, b, tb - ta)
    m = re.search(r"([\d.]+)% idle", top.communicate()[0] or "")
    res["mac_total_busy_pct"] = round(100 - float(m.group(1)), 1) if m else None
    res.update(config=cfg, workload=mode, round=rnd)
    print(json.dumps(res), flush=True)
    return res


def quality_shot(cfg, rnd):
    mac("/tmp/page.sh static")
    time.sleep(5)
    vm(f"DISPLAY=:99 python3 /tmp/bclient.py shot /tmp/q.png")
    dst = f"{OUT}/q_{cfg}_{rnd}.png"
    sh(f"multipass transfer {VM}:/tmp/q.png {XFER}/q.png && cp {XFER}/q.png {dst}")
    if rnd == 0:
        mac("/tmp/shot.sh /tmp/ref_static.png")
        sh(f"scp -q {MAC}:/tmp/ref_static.png {OUT}/ref_static.png")


deploy()
results = []
for rnd in range(ROUNDS):
    for cfg in (CONFIGS if rnd % 2 == 0 else CONFIGS[::-1]):
        vm(f"{SESS_ENV} /tmp/sess.sh {cfg}")
        time.sleep(8)
        quality_shot(cfg, rnd)
        for mode in ["static", "motion", "scroll", "latency", "latencysmall"]:
            results.append(run_workload(cfg, mode, rnd))
            json.dump(results, open(f"{OUT}/results.json", "w"), indent=1)
vm("/tmp/sess.sh stop")
print("DONE")
