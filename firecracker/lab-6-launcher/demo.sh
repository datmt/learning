#!/usr/bin/env bash
# End-to-end demo of launcher.py
set -euo pipefail
cd "$(dirname "$0")"
K=$(ls "$HOME"/fc-lab1/vmlinux-* 2>/dev/null | tail -1 || echo "$HOME/fc-lab1/vmlinux")
R=$(ls "$HOME"/fc-lab1/ubuntu-*.ext4 2>/dev/null | tail -1 || echo "$HOME/fc-lab1/ubuntu.ext4")
[ -f "$K" ] && [ -f "$R" ] || { echo "Run lab-1 fetch first."; exit 1; }
sudo python3 launcher.py boot --kernel "$K" --rootfs "$R" \
  --socket /tmp/fc-demo.socket --tap tap-demo || true
sleep 3
sudo python3 launcher.py status --socket /tmp/fc-demo.socket || true
KEY=$(ls "$HOME"/fc-lab1/ubuntu-*.id_rsa | grep -v pub | tail -1)
ssh -i "$KEY" -o StrictHostKeyChecking=no root@172.16.0.2 \
  "ip addr add 172.16.0.2/30 dev eth0 2>/dev/null; ip link set eth0 up; ip route add default via 172.16.0.1 dev eth0 2>/dev/null; echo nameserver 8.8.8.8 > /etc/resolv.conf; uname -a; echo hello-from-guest" || echo "(guest net not up yet — check TAP/IP)"
echo "Stop with: sudo python3 launcher.py stop --socket /tmp/fc-demo.socket"
