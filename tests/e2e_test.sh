#!/bin/bash
# E2E smoke test for macrdp
# Runs from a Linux machine with FreeRDP 3 against a Mac running macrdp-server.
#
# Usage: MACRDP_USER=<user> MACRDP_PASS=<pass> ./tests/e2e_test.sh <host> [port]
#   or set MACRDP_HOST / MACRDP_PORT instead of arguments (port default: 3389)
#   XFREERDP: FreeRDP client binary (default: xfreerdp3; H.264 support needed)
#   MACRDP_AVC444=1: the server runs with chroma_mode = "avc444"; test 7 then requires AVC444 frames
#   MACRDP_LOG: path of the server log, if readable here (enables the server log check of test 8)

set -euo pipefail

HOST="${1:-${MACRDP_HOST:-}}"
PORT="${2:-${MACRDP_PORT:-3389}}"
USER="${MACRDP_USER:-}"
PASS="${MACRDP_PASS:-}"
XFREERDP="${XFREERDP:-xfreerdp3}"
if [[ -z "$HOST" || -z "$USER" || -z "$PASS" ]]; then
    sed -n '2,9p' "$0" >&2
    exit 2
fi
ADDR="${HOST}:${PORT}"
PASSED=0
FAILED=0
TESTS=()

pass() { PASSED=$((PASSED+1)); TESTS+=("PASS: $1"); echo "  PASS: $1"; }
fail() { FAILED=$((FAILED+1)); TESTS+=("FAIL: $1"); echo "  FAIL: $1"; }

LOGDIR=$(mktemp -d)
trap 'rm -rf "$LOGDIR"' EXIT

# Run a headless session for $1 seconds, client log in $2, extra client arguments after.
# Sets SESSION_RC: 124 means the session was still up when the time ran out.
# The rdpgfx debug filter makes the client log each GFX command ("RecvCapsConfirmPdu",
# "Got GFX RDPGFX_CODECID_<codec>", at most every 500 ms per codec).
session() {
    local secs=$1 log=$2
    shift 2
    SESSION_RC=0
    timeout "$secs" xvfb-run -a "$XFREERDP" /v:"${ADDR}" /u:"${USER}" /p:"${PASS}" /cert:ignore \
        /log-filters:com.freerdp.channels.rdpgfx.client:DEBUG "$@" </dev/null >"$log" 2>&1 \
        || SESSION_RC=$?
}

# Number of GFX surface commands with codec $2 (a prefix, e.g. AVC4) in client log $1.
gfx_frames() { grep -c "Got GFX RDPGFX_CODECID_$2" "$1" || true; }

# First GFX decode or channel error in client log $1 (empty if none).
gfx_error() {
    grep -E "zgfx_decompress failure|EndFrame failed|ChannelId [0-9]+ not found|\[ERROR\]\[com\.freerdp\.(channels\.rdpgfx|codec)" "$1" \
        | head -1 || true
}

# A session that stays up for its whole time, gets GFX caps confirmed and keeps getting
# frames of codec $3 (at least one per second on average) without GFX errors.
# $1: test name, $2: client log, $3: codec prefix, $4: session seconds.
check_gfx_session() {
    local name=$1 log=$2 codec=$3 secs=$4 frames err
    frames=$(gfx_frames "$log" "$codec")
    err=$(gfx_error "$log")
    if [[ $SESSION_RC -ne 124 ]]; then
        fail "$name: session ended early (exit ${SESSION_RC}): $(grep -E 'ERRCONNECT|ERRINFO' "$log" | head -1)"
    elif ! grep -q "RecvCapsConfirmPdu" "$log"; then
        fail "$name: no GFX caps confirm"
    elif [[ -n "$err" ]]; then
        fail "$name: GFX error: ${err}"
    elif [[ $frames -lt $((secs - 5)) ]]; then
        fail "$name: only ${frames} ${codec} frame logs in ${secs}s"
    else
        pass "$name (${frames} ${codec} frame logs)"
    fi
}

echo "=== macrdp E2E tests against ${ADDR} ==="
echo ""

# 1. TCP connectivity
echo "[1] TCP connectivity"
if timeout 3 bash -c "echo > /dev/tcp/${HOST}/${PORT}" 2>/dev/null; then
    pass "TCP connect"
else
    fail "TCP connect"
    echo "Server not reachable — aborting."
    exit 1
fi

# 2. RDP authentication (auth-only mode)
echo "[2] RDP authentication"
AUTH_OUT=$(timeout 10 "$XFREERDP" /v:"${ADDR}" /u:${USER} /p:${PASS} /cert:ignore /auth-only 2>&1) || true
if echo "$AUTH_OUT" | grep -q "Authentication only, exit status 1"; then
    pass "RDP auth succeeds"
else
    fail "RDP auth"
fi

# 3. Wrong password rejected
echo "[3] Wrong password rejected"
BAD_OUT=$(timeout 10 "$XFREERDP" /v:"${ADDR}" /u:${USER} /p:wrong /cert:ignore /auth-only 2>&1) || true
# The server must answer with STATUS_LOGON_FAILURE, as Windows does; mstsc then asks for
# the password again instead of falling back to TLS-only security.
if echo "$BAD_OUT" | grep -q "ERRCONNECT_LOGON_FAILURE"; then
    pass "Bad password rejected with logon failure"
else
    fail "Bad password should be rejected with ERRCONNECT_LOGON_FAILURE"
fi

# 4. Full RDP connection with GFX (headless via xvfb)
echo "[4] Full RDP connection (/gfx: AVC disabled, bitmap updates)"
if command -v xvfb-run &>/dev/null; then
    FULL_OUT=$(timeout 15 xvfb-run -a "$XFREERDP" /v:"${ADDR}" /u:${USER} /p:${PASS} \
        /cert:ignore /gfx /timeout:5000 +auto-reconnect 2>&1) || true
    if echo "$FULL_OUT" | grep -qi "Capabilities\|GFX\|connected\|surface"; then
        pass "GFX connection"
    elif echo "$FULL_OUT" | grep -qi "error\|fail"; then
        fail "GFX connection: $(echo "$FULL_OUT" | grep -i 'error\|fail' | head -1)"
    else
        pass "GFX connection (no errors)"
    fi
else
    echo "  SKIP: xvfb-run not available"
fi

# 5. Session takeover (ConnectionPolicy::Preempt): a second authenticated client gets the
#    session; the first one is told why it was disconnected.
echo "[5] Session takeover"
if command -v xvfb-run &>/dev/null; then
    # FreeRDP does not set RNS_UD_CS_SUPPORT_ERRINFO_PDU by default, and the server must
    # not send a Set Error Info PDU to a client without it (MS-RDPBCGR 3.3.5.7.1).
    A_LOG="$LOGDIR/takeover_a.log"
    timeout 30 xvfb-run -a "$XFREERDP" /v:"${ADDR}" /u:"${USER}" /p:"${PASS}" /cert:ignore \
        /gfx:AVC420 /tune:FreeRDP_SupportErrorInfoPdu:true </dev/null >"$A_LOG" 2>&1 &
    A_PID=$!
    sleep 6

    session 12 "$LOGDIR/takeover_b.log" /gfx:AVC420
    A_DONE=0
    kill -0 $A_PID 2>/dev/null || A_DONE=1
    A_RC=0
    wait $A_PID 2>/dev/null || A_RC=$?

    if [[ $A_DONE -ne 1 ]]; then
        fail "Takeover: first session still running after the second connected"
    elif ! grep -q "ERRINFO_DISCONNECTED_BY_OTHER_CONNECTION" "$A_LOG"; then
        fail "Takeover: first session ended without ERRINFO_DISCONNECTED_BY_OTHER_CONNECTION (exit ${A_RC})"
    else
        pass "Takeover: first session got ERRINFO_DISCONNECTED_BY_OTHER_CONNECTION"
    fi
    check_gfx_session "Takeover: second session" "$LOGDIR/takeover_b.log" AVC420 12
    sleep 2
else
    echo "  SKIP: xvfb-run not available"
fi

# 6. TLS-only security (no NLA) with an empty password must not get a session
echo "[6] TLS-only, empty password rejected"
if command -v xvfb-run &>/dev/null; then
    # A session stays up until timeout ends the client (exit 124); a rejected one ends early.
    TLS_RC=0
    TLS_OUT=$(timeout 15 xvfb-run -a "$XFREERDP" /v:"${ADDR}" /u:${USER} /p: /sec:tls \
        /cert:ignore /gfx /timeout:5000 </dev/null 2>&1) || TLS_RC=$?
    if [[ $TLS_RC -ne 124 ]] && echo "$TLS_OUT" | grep -q "ERRCONNECT"; then
        pass "TLS-only empty password rejected"
    else
        fail "TLS-only empty password should be rejected"
    fi
else
    echo "  SKIP: xvfb-run not available"
fi

# 7. AVC444 client
echo "[7] AVC444 client"
if command -v xvfb-run &>/dev/null; then
    # A server with chroma_mode = "avc420" sends AVC420 to an AVC444 client.
    # 1280x720 (not the 1024x768 default) gives GFX PDUs larger than 64 KiB.
    CODEC=AVC4
    [[ "${MACRDP_AVC444:-}" == 1 ]] && CODEC=AVC444
    session 20 "$LOGDIR/avc444.log" /gfx:AVC444 /size:1280x720
    check_gfx_session "AVC444 client" "$LOGDIR/avc444.log" "$CODEC" 20
    sleep 2
else
    echo "  SKIP: xvfb-run not available"
fi

# 8. Client asks for no audio (/audio-mode:2): rdpsnd is not started
echo "[8] No audio (/audio-mode:2)"
if command -v xvfb-run &>/dev/null; then
    LOG_START=0
    if [[ -n "${MACRDP_LOG:-}" && -r "${MACRDP_LOG}" ]]; then
        LOG_START=$(wc -l <"$MACRDP_LOG")
    fi
    session 10 "$LOGDIR/noaudio.log" /gfx:AVC420 /audio-mode:2
    check_gfx_session "No audio: session" "$LOGDIR/noaudio.log" AVC420 10
    if [[ -n "${MACRDP_LOG:-}" && -r "${MACRDP_LOG}" ]]; then
        if tail -n +"$((LOG_START + 1))" "$MACRDP_LOG" | grep -q "rdpsnd not started"; then
            pass "No audio: server did not start rdpsnd"
        else
            fail "No audio: server log has no \"rdpsnd not started\""
        fi
    else
        echo "  SKIP: server log check (MACRDP_LOG not set or not readable)"
    fi
    sleep 2
else
    echo "  SKIP: xvfb-run not available"
fi

# 9. Dynamic resolution client
echo "[9] Dynamic resolution (/dynamic-resolution)"
if command -v xvfb-run &>/dev/null; then
    session 10 "$LOGDIR/dynres.log" /gfx:AVC420 /dynamic-resolution
    check_gfx_session "Dynamic resolution" "$LOGDIR/dynres.log" AVC420 10
else
    echo "  SKIP: xvfb-run not available"
fi

# Summary
echo ""
echo "=== Results: ${PASSED} passed, ${FAILED} failed ==="
for t in "${TESTS[@]}"; do echo "  $t"; done

exit ${FAILED}
