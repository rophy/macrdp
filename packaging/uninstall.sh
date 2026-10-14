#!/bin/sh
# Stop and remove macrdp-server. Keeps config, TLS certificate and logs.
set -eu

LABEL=io.github.rophy.macrdp
PREFIX=${PREFIX:-$HOME/.local}

launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
rm -f "$HOME/Library/LaunchAgents/$LABEL.plist" "$PREFIX/bin/macrdp-server"

cat <<EOF
Removed macrdp-server.
Kept: $HOME/.config/macrdp (config, TLS certificate), $HOME/Library/Logs/macrdp
Remove macrdp-server from System Settings > Privacy & Security yourself.
EOF
