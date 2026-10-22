#!/bin/bash
# Install + (re)load the macrdp LaunchAgent for the current user.
#
# Seeds ~/Library/Application Support/macrdp/config.env from the example on
# first run, renders the LaunchAgent plist from the template, and bootstraps it.
#
# Env overrides:
#   APP_DIR=/Applications     # where macrdp.app was installed (default /Applications)
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PKG_DIR="$REPO_ROOT/packaging"
APP_DIR="${APP_DIR:-/Applications}"
# MUST match the BUNDLE_PREFIX used by make-app.sh (and gui/make-tray-app.sh),
# or the controller targets a different label than the agent installed here.
BUNDLE_PREFIX="${BUNDLE_PREFIX:-com.clintcan}"
LABEL="$BUNDLE_PREFIX.macrdp"
UID_NUM="$(id -u)"

APP="$APP_DIR/macrdp.app"
[ -d "$APP" ] || { echo "macrdp.app not found at $APP — run packaging/make-app.sh first" >&2; exit 1; }

# 1. Seed config.env if absent.
SUPPORT="$HOME/Library/Application Support/macrdp"
mkdir -p "$SUPPORT" "$HOME/Library/Logs" "$HOME/Library/LaunchAgents"
CONFIG="$SUPPORT/config.env"
if [ ! -f "$CONFIG" ]; then
    cp "$PKG_DIR/config.env.example" "$CONFIG"
    echo "==> seeded $CONFIG (edit to taste)"
else
    echo "==> keeping existing $CONFIG"
    # Earlier versions wrote ENABLE_H264=0 / ADAPTIVE_BITRATE=0 into config.env; the kept file
    # keeps them (the server also warns at start). Say so, so an upgrade does not stay on bitmaps.
    for key in ENABLE_H264 ADAPTIVE_BITRATE; do
        if grep -Eq "^[[:space:]]*(export[[:space:]]+)?$key=[\"']?0" "$CONFIG"; then
            echo "NOTE: $CONFIG sets $key=0 (the default of earlier versions; now on by default)."
            echo "      Delete that line or set $key=1 to use the new default."
        fi
    done
fi

# 2. Render the LaunchAgent plist from the template.
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
sed -e "s#__LABEL__#$LABEL#g" -e "s#__APP_DIR__#$APP_DIR#g" -e "s#__HOME__#$HOME#g" \
    "$PKG_DIR/launchagent.plist.template" > "$PLIST"
echo "==> wrote $PLIST"

# 3. (Re)bootstrap the agent. `bootstrap` immediately after `bootout` can fail
#    with "Input/output error" (EIO, 5) while launchd is still tearing the old
#    job down — a race that otherwise leaves the agent UNloaded (server doesn't
#    come back). Retry a few times; the final attempt runs without suppressing
#    stderr so a genuine failure is surfaced (and aborts via set -e).
launchctl bootout "gui/$UID_NUM/$LABEL" 2>/dev/null || true
bootstrapped=0
for _ in 1 2 3 4 5; do
    if launchctl bootstrap "gui/$UID_NUM" "$PLIST" 2>/dev/null; then
        bootstrapped=1
        break
    fi
    sleep 1
done
[ "$bootstrapped" = 1 ] || launchctl bootstrap "gui/$UID_NUM" "$PLIST"
launchctl enable "gui/$UID_NUM/$LABEL"
launchctl kickstart -k "gui/$UID_NUM/$LABEL"

echo
echo "Loaded $LABEL."
echo "  status:  launchctl print gui/$UID_NUM/$LABEL | grep -E 'state|pid'"
echo "  logs:    tail -f ~/Library/Logs/macrdp.log"
echo "  apply config change:  launchctl kickstart -k gui/$UID_NUM/$LABEL"
echo "  stop:    launchctl bootout gui/$UID_NUM/$LABEL"
