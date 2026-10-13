#!/usr/bin/env python3
"""Count rectangle encodings in a server->client RFB stream dump (starts at ServerInit, 32bpp)."""
import struct, sys, zlib
from collections import Counter
b = open(sys.argv[1], "rb").read()
p = 0
def take(n):
    global p
    if p + n > len(b): raise EOFError
    r = b[p:p+n]; p += n; return r
def compact():
    n = 0
    for i in range(3):
        c = take(1)[0]; n |= (c & 0x7f) << (7 * i)
        if not c & 0x80: break
    return n
w, h = struct.unpack(">HH", take(4)); take(16); take(struct.unpack(">I", take(4))[0])
encs, tight, bytes_by, updates = Counter(), Counter(), Counter(), 0
try:
    while True:
        t = take(1)[0]
        if t == 0:
            take(1); n = struct.unpack(">H", take(2))[0]; updates += 1
            for _ in range(n):
                x, y, rw, rh, e = struct.unpack(">HHHHi", take(12)); s0 = p
                encs[e] += 1
                if e == 0: take(rw * rh * 4)
                elif e == 1: take(4)
                elif e == 7:
                    cc = take(1)[0]; kind = cc >> 4
                    if kind == 8: take(3); tight["fill"] += 1
                    elif kind == 9: take(compact()); tight["jpeg"] += 1
                    else:
                        filt = take(1)[0] if cc & 0x40 else 0
                        if filt == 1:
                            nc = take(1)[0] + 1; take(nc * 3); size = ((rw + 7) // 8) * rh if nc == 2 else rw * rh
                            tight["palette"] += 1
                        else:
                            size = rw * rh * 3; tight["gradient" if filt == 2 else "basic-zlib"] += 1
                        take(size) if size < 12 else take(compact())
                elif e in (16, 6): take(struct.unpack(">I", take(4))[0])
                elif e == -239: take(rw * rh * 4 + ((rw + 7) // 8) * rh)
                elif e in (-223, -224, -232): 
                    if e == -224: bytes_by[e] += 0; break
                elif e == -308:
                    nscr = take(1)[0]; take(3); take(16 * nscr)
                else:
                    print("unknown encoding", e, "at", p, "- stop"); raise EOFError
                bytes_by[e] += p - s0
        elif t == 3: take(3); take(struct.unpack(">I", take(4))[0])
        elif t == 2: pass
        else:
            print("unknown server msg", t, "at", p); break
except EOFError:
    pass
print(f"fb {w}x{h} updates={updates} parsed={p}/{len(b)} bytes")
print("rects by encoding:", dict(encs)); print("bytes by encoding:", dict(bytes_by)); print("tight subtypes:", dict(tight))
