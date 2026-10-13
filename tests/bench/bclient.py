#!/usr/bin/env python3
"""Client-side measurements on the Xvfb display that shows the remote desktop 1:1.
usage:
  bclient.py latency <trials> <out.json>      key -> patch color flip at (200,300)
  bclient.py fps <seconds> <out.json>         read Gray-code frame counter strip
  bclient.py shot <out.png>                   grab full screen
Every mode also records NIC rx/tx bytes over the run."""
import json, random, statistics, sys, time
from Xlib import X, display
from Xlib.ext import xtest

d = display.Display()
root = d.screen().root
IFACE = "ens3"


def nic():
    r = int(open(f"/sys/class/net/{IFACE}/statistics/rx_bytes").read())
    t = int(open(f"/sys/class/net/{IFACE}/statistics/tx_bytes").read())
    return r, t


def pixel(x, y):
    raw = root.get_image(x, y, 1, 1, X.ZPixmap, 0xffffffff).data
    return raw[2], raw[1], raw[0]  # BGRX -> RGB


def lum(rgb):
    return sum(rgb) / 3


def key_tap(keycode):
    xtest.fake_input(d, X.KeyPress, keycode)
    d.sync()
    xtest.fake_input(d, X.KeyRelease, keycode)
    d.sync()


def stats(v):
    v = sorted(v)
    if not v:
        return {}
    q = lambda p: v[min(len(v) - 1, int(p * len(v)))]
    return {"n": len(v), "mean": round(statistics.mean(v), 2), "p50": round(q(0.5), 2),
            "p90": round(q(0.9), 2), "p99": round(q(0.99), 2), "min": round(v[0], 2), "max": round(v[-1], 2)}


def latency(trials):
    kc = d.keysym_to_keycode(ord("a"))
    res, timeouts = [], 0
    for i in range(trials):
        before = lum(pixel(200, 300)) > 128
        t0 = time.perf_counter()
        key_tap(kc)
        while True:
            now = time.perf_counter()
            if (lum(pixel(200, 300)) > 128) != before:
                res.append((now - t0) * 1000)
                break
            if now - t0 > 3:
                timeouts += 1
                break
            time.sleep(0.0002)
        time.sleep(random.uniform(0.4, 0.9))
    return {"latency_ms": stats(res), "timeouts": timeouts, "raw": [round(x, 2) for x in res]}


BITS, BS, CX, CY = 12, 40, 1280 - 12 * 40 - 20, 40


def read_counter():
    raw = root.get_image(CX, CY + BS // 2, BITS * BS, 1, X.ZPixmap, 0xffffffff).data
    g = 0
    for i in range(BITS):
        off = (i * BS + BS // 2) * 4
        g = (g << 1) | (1 if (raw[off] + raw[off + 1] + raw[off + 2]) / 3 > 128 else 0)
    n = 0
    while g:
        n ^= g
        g >>= 1
    return n


def fps(seconds, stable_ms=3.0):
    # A value counts as a shown frame only after it stays unchanged for stable_ms:
    # tile-based updates can leave the strip torn between two frames for a moment.
    seen, cand, cand_t, last = [], None, 0.0, None
    t_end = time.perf_counter() + seconds
    reads = 0
    while time.perf_counter() < t_end:
        v = read_counter()
        now = time.perf_counter()
        reads += 1
        time.sleep(0.0005)
        if v != cand:
            cand, cand_t = v, now
        elif v != last and (now - cand_t) * 1000 >= stable_ms:
            seen.append((cand_t, v))
            last = v
    if len(seen) < 3:
        return {"frames": len(seen), "reads": reads}
    seen = seen[1:]  # first value existed before the window
    dur = seen[-1][0] - seen[0][0]
    gaps = [(b[0] - a[0]) * 1000 for a, b in zip(seen, seen[1:])]
    steps = [(b[1] - a[1]) % 4096 for a, b in zip(seen, seen[1:])]
    fwd = [x for x in steps if x < 2048]
    src = sum(fwd)  # source frames rendered by the page across the window
    return {"shown_fps": round((len(seen) - 1) / dur, 2), "source_fps": round(src / dur, 2),
            "delivered_ratio": round(len(fwd) / max(src, 1), 3),
            "gap_ms": stats(gaps), "big_gaps_over_100ms": sum(1 for g in gaps if g > 100),
            "bad_reads": len(steps) - len(fwd), "sampler_reads_per_s": round(reads / seconds)}


def shot(path):
    from PIL import Image
    g = root.get_geometry()
    raw = root.get_image(0, 0, g.width, g.height, X.ZPixmap, 0xffffffff).data
    Image.frombytes("RGB", (g.width, g.height), raw, "raw", "BGRX").save(path)
    return {"saved": path}


mode = sys.argv[1]
r0, t0, w0 = *nic(), time.time()
if mode == "latency":
    out = latency(int(sys.argv[2]))
elif mode == "fps":
    out = fps(float(sys.argv[2]))
else:
    out = shot(sys.argv[2])
r1, t1, w1 = *nic(), time.time()
out["rx_mbps"] = round((r1 - r0) * 8 / (w1 - w0) / 1e6, 3)
out["tx_mbps"] = round((t1 - t0) * 8 / (w1 - w0) / 1e6, 3)
out["rx_MB"] = round((r1 - r0) / 1e6, 2)
out["wall_s"] = round(w1 - w0, 2)
if mode != "shot":
    json.dump(out, open(sys.argv[3], "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "raw"}))
