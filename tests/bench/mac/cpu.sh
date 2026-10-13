#!/bin/sh
# Print "<pid> <cpu-seconds> <comm>" for processes involved in remote desktop.
ps -A -o pid=,time=,comm= | grep -E "macrdp-server|screensharingd|ScreensharingAgent|SSMenuAgent|replayd|WindowServer" | grep -v grep \
  | awk '{ n=split($2,a,":"); s=0; for(i=1;i<=n;i++) s=s*60+a[i]; print $1, s, $NF }'
