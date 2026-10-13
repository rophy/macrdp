#!/usr/bin/env python3
"""PSNR / SSIM of client screenshots vs the Mac's own screenshot of the static page.
usage: quality.py ref.png a.png [b.png ...]"""
import json, sys
import numpy as np
from PIL import Image


def load(p):
    return np.asarray(Image.open(p).convert("RGB"), dtype=np.float64)


def psnr(a, b):
    mse = np.mean((a - b) ** 2)
    return 99.0 if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def box(x, k=8):
    c = np.cumsum(np.cumsum(np.pad(x, ((1, 0), (1, 0))), 0), 1)
    return (c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)


def ssim(a, b):
    ga = a @ [0.299, 0.587, 0.114]
    gb = b @ [0.299, 0.587, 0.114]
    C1, C2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    ma, mb = box(ga), box(gb)
    va, vb, cov = box(ga * ga) - ma ** 2, box(gb * gb) - mb ** 2, box(ga * gb) - ma * mb
    s = ((2 * ma * mb + C1) * (2 * cov + C2)) / ((ma ** 2 + mb ** 2 + C1) * (va + vb + C2))
    return float(s.mean())


# Regions of the static test page (see bench.html)
REGIONS = {"full": (0, 720), "photo": (0, 360), "text": (360, 575), "colorbars": (575, 720)}
ref = load(sys.argv[1])
out = {}
for p in sys.argv[2:]:
    img = load(p)
    out[p.rsplit("/", 1)[-1]] = {
        r: {"psnr_db": round(psnr(ref[y0:y1], img[y0:y1]), 2), "ssim": round(ssim(ref[y0:y1], img[y0:y1]), 4)}
        for r, (y0, y1) in REGIONS.items()}
print(json.dumps(out, indent=1))
