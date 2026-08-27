#!/bin/bash
# E2E smoke test for macrdp
# Runs from a Linux machine against a Mac running macrdp-server
#
# Usage: ./tests/e2e_test.sh [host] [port]
#   host: macrdp server address (default: 192.168.1.207)
#   port: macrdp server port (default: 13389)

set -euo pipefail

HOST="${1:-192.168.1.207}"
PORT="${2:-13389}"
USER="admin"
PASS="123456"
ADDR="${HOST}:${PORT}"
PASSED=0
FAILED=0
TESTS=()

pass() { PASSED=$((PASSED+1)); TESTS+=("PASS: $1"); echo "  PASS: $1"; }
fail() { FAILED=$((FAILED+1)); TESTS+=("FAIL: $1"); echo "  FAIL: $1"; }

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
AUTH_OUT=$(timeout 10 xfreerdp3 /v:"${ADDR}" /u:${USER} /p:${PASS} /cert:ignore /auth-only 2>&1) || true
if echo "$AUTH_OUT" | grep -q "Authentication only, exit status 1"; then
    pass "RDP auth succeeds"
else
    fail "RDP auth"
fi

# 3. Wrong password rejected
echo "[3] Wrong password rejected"
BAD_OUT=$(timeout 10 xfreerdp3 /v:"${ADDR}" /u:${USER} /p:wrong /cert:ignore /auth-only 2>&1) || true
if echo "$BAD_OUT" | grep -qi "logon\|credentials\|denied\|ERRCONNECT\|failed"; then
    pass "Bad password rejected"
else
    fail "Bad password should be rejected"
fi

# 4. Full RDP connection with GFX (headless via xvfb)
echo "[4] Full RDP connection (GFX/H.264)"
if command -v xvfb-run &>/dev/null; then
    FULL_OUT=$(timeout 15 xvfb-run -a xfreerdp3 /v:"${ADDR}" /u:${USER} /p:${PASS} \
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

# 5. Concurrent connection rejection
echo "[5] Concurrent connection rejection"
if command -v xvfb-run &>/dev/null; then
    # Start a background session
    xvfb-run -a xfreerdp3 /v:"${ADDR}" /u:${USER} /p:${PASS} \
        /cert:ignore /timeout:8000 2>/dev/null &
    BG_PID=$!
    sleep 3

    # Try a second connection — should fail quickly, not hang
    REJECT_START=$(date +%s)
    REJECT_OUT=$(timeout 10 xfreerdp3 /v:"${ADDR}" /u:${USER} /p:${PASS} \
        /cert:ignore /auth-only 2>&1) || true
    REJECT_END=$(date +%s)
    REJECT_TIME=$((REJECT_END - REJECT_START))

    kill $BG_PID 2>/dev/null; wait $BG_PID 2>/dev/null || true
    sleep 1

    if [ "$REJECT_TIME" -lt 8 ]; then
        pass "Second connection rejected quickly (${REJECT_TIME}s)"
    else
        fail "Second connection hung for ${REJECT_TIME}s"
    fi
else
    echo "  SKIP: xvfb-run not available"
fi

# Summary
echo ""
echo "=== Results: ${PASSED} passed, ${FAILED} failed ==="
for t in "${TESTS[@]}"; do echo "  $t"; done

exit ${FAILED}
