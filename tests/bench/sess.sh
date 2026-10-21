#!/bin/bash
# Runs on the client VM. usage: sess.sh stop | rdp
# Env: MAC_HOST, RDP_PORT (default 3390), RDP_USER, RDP_PASS_FILE (file on this VM that holds the
# password; it goes to FreeRDP on stdin, never on a command line), BENCH_DISPLAY (default :99)
export DISPLAY=${BENCH_DISPLAY:-:99}
PIDFILE=/tmp/macrdp-bench-rdp.pid
stop() {
  [ -f $PIDFILE ] && kill "$(cat $PIDFILE)" 2>/dev/null
  rm -f $PIDFILE
  sleep 1
}
case "$1" in
  stop) stop ;;
  rdp)
    stop
    { printf '%s\n' "/v:$MAC_HOST:${RDP_PORT:-3390}" "/u:$RDP_USER"
      printf '/p:%s\n' "$(head -n 1 "$RDP_PASS_FILE")"
      printf '%s\n' /cert:ignore /f /gfx:AVC420; } \
      | setsid nohup /usr/local/bin/xfreerdp /args-from:stdin >/tmp/macrdp-bench-rdp.log 2>&1 &
    echo $! >$PIDFILE
    ;;
esac
