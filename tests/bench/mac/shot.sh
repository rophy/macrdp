#!/bin/sh
# usage: shot.sh <out.png>
# Captures via launchd in the GUI session: screencapture over SSH sees no windows.
# Needs a Screen Recording TCC grant for /usr/sbin/screencapture.
launchctl remove com.bench.shot 2>/dev/null
rm -f "$1"
launchctl submit -l com.bench.shot -- /usr/sbin/screencapture -x "$1"
for i in 1 2 3 4 5 6 7 8 9 10; do [ -s "$1" ] && break; sleep 0.5; done
launchctl remove com.bench.shot 2>/dev/null
