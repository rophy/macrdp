#!/bin/sh
# Install macrdp-server for the current user and start it as a LaunchAgent.
# Usage: ./install.sh            (binary goes to ~/.local/bin)
#        PREFIX=/opt/macrdp ./install.sh
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
LABEL=io.github.rophy.macrdp
PREFIX=${PREFIX:-$HOME/.local}
BIN=$PREFIX/bin/macrdp-server
CONF_DIR=$HOME/.config/macrdp
CONF=$CONF_DIR/config.toml
LOG_DIR=$HOME/Library/Logs/macrdp
PLIST=$HOME/Library/LaunchAgents/$LABEL.plist
DOMAIN=gui/$(id -u)

mkdir -p "$PREFIX/bin" "$CONF_DIR" "$LOG_DIR" "$HOME/Library/LaunchAgents"
install -m 755 "$HERE/macrdp-server" "$BIN"

if [ -f "$CONF" ]; then
    echo "Kept existing $CONF"
else
    user=$(id -un)
    pass=$(LC_ALL=C tr -dc 'A-Za-z0-9' </dev/urandom | head -c 16)
    sed -e "s/^username = .*/username = \"$user\"/" \
        -e "s/^password = .*/password = \"$pass\"/" \
        "$HERE/config.example.toml" > "$CONF"
    chmod 600 "$CONF"
    echo "Created $CONF"
    echo "RDP login: $user / $pass"
fi

sed -e "s|@BIN@|$BIN|g" -e "s|@CONF@|$CONF|g" -e "s|@LOG_DIR@|$LOG_DIR|g" \
    "$HERE/$LABEL.plist" > "$PLIST"
launchctl bootout "$DOMAIN/$LABEL" 2>/dev/null || true
launchctl bootstrap "$DOMAIN" "$PLIST"

port=$(sed -n 's/^port = \([0-9]*\).*/\1/p' "$CONF")
cat <<EOF

macrdp-server is running on port ${port:-3389}. Logs: $LOG_DIR

Give $BIN these permissions in
System Settings > Privacy & Security:
  - Screen Recording
  - Accessibility
Then restart it:
  launchctl kickstart -k $DOMAIN/$LABEL
EOF
