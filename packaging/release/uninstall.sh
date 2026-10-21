#!/bin/sh
# Stop and remove macrdp installed by install.sh. Keeps config, TLS certificate, Keychain entry
# and logs.
set -eu

LABEL=${BUNDLE_PREFIX:-com.clintcan}.macrdp
PREFIX=${PREFIX:-$HOME/.local}
BIN=$PREFIX/bin/macrdp
PLIST=$HOME/Library/LaunchAgents/$LABEL.plist

# The .app install (packaging/install-launchagent.sh) uses the same label: leave its agent alone.
if [ -f "$PLIST" ] && ! grep -qF "<string>$BIN</string>" "$PLIST"; then
    echo "$PLIST does not run $BIN (the macrdp.app agent?); not removed."
else
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"
fi
rm -f "$BIN"

cat <<MSG
Removed $BIN.
Kept: $HOME/Library/Application Support/macrdp (config.env, TLS certificate),
      $HOME/Library/Logs/macrdp.log, the Keychain entry (security delete-generic-password -s macrdp)
Remove macrdp from System Settings > Privacy & Security yourself.
MSG
