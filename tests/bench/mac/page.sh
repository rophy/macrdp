#!/bin/sh
# usage: page.sh <mode> | close  — show bench.html#<mode> (next to this script) in a full-screen
# Chrome kiosk window with its own profile in the same folder; "close" quits that window.
D=$(cd "$(dirname "$0")" && pwd)
pkill -f "user-data-dir=$D/chrome"; sleep 1
[ "$1" = close ] && exit 0
open -na "Google Chrome" --args --kiosk --start-fullscreen --user-data-dir="$D/chrome" \
  --no-first-run --no-default-browser-check "file://$D/bench.html#$1"
sleep 4
