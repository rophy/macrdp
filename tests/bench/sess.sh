#!/bin/bash
# Runs on the client VM. usage: sess.sh stop | rdp | vnc | vncjpeg
# Env: MAC_HOST, RDP_PORT, RDP_USER, RDP_PASS, VNC_USER, VNC_PASS (macOS account for Apple auth)
export DISPLAY=:99
stop() {
  pkill -x xfreerdp; pkill -x xtigervncviewer
  pkill -f "python3 /tmp/ardproxy.py"
  sleep 1
}
vnc() {
  stop
  S2C_DUMP=/tmp/s2c_$1.bin setsid nohup python3 /tmp/ardproxy.py "$MAC_HOST" 5900 5999 "$VNC_USER" "$VNC_PASS" >/tmp/proxy.log 2>&1 < /dev/null &
  sleep 1
  shift
  setsid nohup xtigervncviewer -FullScreen -SecurityTypes None "$@" 127.0.0.1::5999 >/tmp/vnc.log 2>&1 < /dev/null &
}
case "$1" in
  stop) stop ;;
  rdp)
    stop
    setsid nohup /usr/local/bin/xfreerdp /v:"$MAC_HOST:${RDP_PORT:-3389}" /u:"$RDP_USER" /p:"$RDP_PASS" /cert:ignore /f /gfx:AVC420 >/tmp/rdp.log 2>&1 < /dev/null &
    ;;
  vnc) vnc vnc ;;
  vncjpeg) vnc vncjpeg -AutoSelect=0 -FullColor=1 -PreferredEncoding=Tight -QualityLevel=6 -CompressLevel=2 ;;
esac
