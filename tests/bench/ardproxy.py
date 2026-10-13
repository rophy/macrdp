#!/usr/bin/env python3
"""RFB proxy: Apple DH auth (type 30) upstream, 'None' auth downstream, then raw relay.
Logs the client's SetEncodings list."""
import hashlib, os, socket, struct, sys, threading
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

UP_HOST, UP_PORT = sys.argv[1], int(sys.argv[2])
LISTEN_PORT = int(sys.argv[3])
USER, PASS = sys.argv[4].encode(), sys.argv[5].encode()


def recvn(s, n):
    b = b""
    while len(b) < n:
        c = s.recv(n - len(b))
        if not c:
            raise EOFError
        b += c
    return b


def upstream_auth():
    u = socket.create_connection((UP_HOST, UP_PORT))
    u.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    ver = recvn(u, 12)
    print("server version", ver, flush=True)
    u.sendall(b"RFB 003.008\n")
    n = recvn(u, 1)[0]
    types = list(recvn(u, n))
    print("server sec types", types, flush=True)
    if 30 not in types:
        raise RuntimeError("no ARD auth")
    u.sendall(bytes([30]))
    gen, keylen = struct.unpack(">HH", recvn(u, 4))
    prime = int.from_bytes(recvn(u, keylen), "big")
    spub = int.from_bytes(recvn(u, keylen), "big")
    priv = int.from_bytes(os.urandom(keylen), "big") % (prime - 2) + 1
    cpub = pow(gen, priv, prime)
    shared = pow(spub, priv, prime).to_bytes(keylen, "big")
    key = hashlib.md5(shared).digest()
    creds = bytearray(os.urandom(128))
    creds[0:len(USER) + 1] = USER + b"\0"
    creds[64:64 + len(PASS) + 1] = PASS + b"\0"
    enc = Cipher(algorithms.AES(key), modes.ECB()).encryptor()
    ct = enc.update(bytes(creds)) + enc.finalize()
    u.sendall(ct + cpub.to_bytes(keylen, "big"))
    res = struct.unpack(">I", recvn(u, 4))[0]
    if res != 0:
        raise RuntimeError("auth failed %d" % res)
    print("upstream auth ok", flush=True)
    return u


DUMP = os.environ.get("S2C_DUMP")


def pipe(a, b, tag):
    f = open(DUMP, "wb") if DUMP and tag == "s2c" else None
    try:
        while True:
            d = a.recv(65536)
            if not d:
                break
            b.sendall(d)
            if f:
                f.write(d)
    except OSError:
        pass
    for s in (a, b):
        try:
            s.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def log_client(c, u):
    """Relay client->server, parsing messages to log SetEncodings."""
    sizes = {0: 20, 3: 10, 4: 8, 5: 6}
    buf = b""
    try:
        # ClientInit (1 byte) handled by caller
        while True:
            t = recvn(c, 1)
            if t[0] == 2:
                hdr = recvn(c, 3)
                cnt = struct.unpack(">xH", hdr)[0]
                body = recvn(c, 4 * cnt)
                encs = struct.unpack(">%di" % cnt, body)
                print("SetEncodings", encs, flush=True)
                u.sendall(t + hdr + body)
            elif t[0] == 6:
                hdr = recvn(c, 7)
                ln = struct.unpack(">3xI", hdr)[0]
                u.sendall(t + hdr + recvn(c, ln))
            elif t[0] in sizes:
                u.sendall(t + recvn(c, sizes[t[0]] - 1))
            else:
                # unknown message: fall back to raw relay
                print("unknown client msg", t[0], "-> raw relay", flush=True)
                u.sendall(t)
                pipe(c, u, "c2s")
                return
    except (EOFError, OSError):
        pass
    for s in (c, u):
        try:
            s.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def handle(c):
    c.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    u = upstream_auth()
    c.sendall(b"RFB 003.008\n")
    recvn(c, 12)
    c.sendall(bytes([1, 1]))  # one type: None
    recvn(c, 1)
    c.sendall(struct.pack(">I", 0))
    u.sendall(recvn(c, 1))  # ClientInit shared flag
    threading.Thread(target=pipe, args=(u, c, "s2c"), daemon=True).start()
    log_client(c, u)


ls = socket.socket()
ls.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
ls.bind(("127.0.0.1", LISTEN_PORT))
ls.listen(4)
print("listening", LISTEN_PORT, flush=True)
while True:
    c, _ = ls.accept()
    threading.Thread(target=handle, args=(c,), daemon=True).start()
