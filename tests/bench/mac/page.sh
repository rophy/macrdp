#!/bin/sh
# usage: page.sh <mode>  — show /tmp/bench.html#<mode> in a full-screen Chrome kiosk window
pkill -f "user-data-dir=/tmp/chromebench"; sleep 1
open -na "Google Chrome" --args --kiosk --start-fullscreen --user-data-dir=/tmp/chromebench \
  --no-first-run --no-default-browser-check "file:///tmp/bench.html#$1"
sleep 4
