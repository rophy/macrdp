#!/bin/sh
# usage: cpu.sh <rdp-port>
# Print "<pid> <cpu-seconds> <name>" for the macrdp process that listens on <rdp-port> (name
# "macrdp") and for the macOS processes involved in screen capture.
pid=$(lsof -t -nP -iTCP:"$1" -sTCP:LISTEN | head -1)
{
  [ -n "$pid" ] && ps -o pid=,time= -p "$pid" | awk '{ print $0, "macrdp" }'
  ps -A -o pid=,time=,comm= | grep -E "replayd|WindowServer" | grep -v grep
} | awk '{ n=split($2,a,":"); s=0; for(i=1;i<=n;i++) s=s*60+a[i]; print $1, s, $NF }'
