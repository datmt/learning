#!/usr/bin/env bash
set -euo pipefail
BR="${BR:-fc-br0}"
for t in tap-a tap-b tap-c; do sudo ip link del "$t" 2>/dev/null || true; done
sudo ip link set dev "$BR" down 2>/dev/null || true
sudo ip link del "$BR" 2>/dev/null || true
for s in /tmp/fc-a.socket /tmp/fc-b.socket /tmp/fc-c.socket; do sudo rm -f "$s"; done
echo "Bridge + TAPs removed."
