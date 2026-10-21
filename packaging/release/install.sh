#!/bin/sh
# Install macrdp from this release tarball for the current user and start it as a LaunchAgent.
# Usage: ./install.sh            (binary goes to ~/.local/bin)
#        PREFIX=/opt/macrdp ./install.sh
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
# Same label as packaging/install-launchagent.sh (the .app install), so the two replace each
# other instead of both binding the port; the menu-bar controller drives this label.
LABEL=${BUNDLE_PREFIX:-com.clintcan}.macrdp
PREFIX=${PREFIX:-$HOME/.local}
BIN=$PREFIX/bin/macrdp
SUPPORT="$HOME/Library/Application Support/macrdp"
CONF=$SUPPORT/config.env
PLIST=$HOME/Library/LaunchAgents/$LABEL.plist
DOMAIN=gui/$(id -u)
user=$(id -un)

mkdir -p "$PREFIX/bin" "$SUPPORT" "$HOME/Library/Logs" "$HOME/Library/LaunchAgents"
install -m 755 "$HERE/macrdp" "$BIN"

if [ -f "$CONF" ]; then
    echo "Kept existing $CONF"
else
    install -m 600 "$HERE/config.env.example" "$CONF"
    echo "Created $CONF (listens on 127.0.0.1:3390; set BIND there for the LAN)"
fi

# The server checks the RDP login against this Mac account (PAM); launchd starts it without a
# terminal, so it reads the account password from the Keychain (USE_KEYCHAIN=1 in config.env).
if security find-generic-password -s macrdp -a "$user" >/dev/null 2>&1; then
    echo "Kept the existing Keychain entry (service macrdp, account $user)"
elif [ -t 0 ]; then
    echo "Store the macOS password of $user in the Keychain (service macrdp):"
    # -w last: security prompts for the password, so it is not on a command line.
    security add-generic-password -U -s macrdp -a "$user" -w
else
    echo "No Keychain entry. Run: security add-generic-password -s macrdp -a $user -w  (it prompts)"
fi

if launchctl print "$DOMAIN/com.user.macrdp" >/dev/null 2>&1; then
    echo "WARNING: the dist/install.sh agent com.user.macrdp is also loaded and uses the same port."
    echo "         Stop it with: launchctl bootout $DOMAIN/com.user.macrdp"
fi

sed -e "s#__LABEL__#$LABEL#g" -e "s#__BIN__#$BIN#g" -e "s#__CONF__#$CONF#g" -e "s#__HOME__#$HOME#g" \
    "$HERE/launchagent.plist.template" > "$PLIST"
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
# bootstrap right after bootout can fail with EIO while launchd tears the old job down.
for _ in 1 2 3 4; do
    launchctl bootstrap "$DOMAIN" "$PLIST" 2>/dev/null && break
    sleep 1
done
launchctl print "$DOMAIN/$LABEL" >/dev/null 2>&1 || launchctl bootstrap "$DOMAIN" "$PLIST"

cat <<MSG

macrdp is running as $LABEL. Config: $CONF
Logs: $HOME/Library/Logs/macrdp.log

Give $BIN these permissions in
System Settings > Privacy & Security:
  - Screen Recording
  - Accessibility
Then restart it:
  launchctl kickstart -k $DOMAIN/$LABEL
MSG
