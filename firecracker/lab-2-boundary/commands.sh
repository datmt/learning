#!/usr/bin/env bash
# Lab 2 runbook.
set -u
API_SOCKET="${API_SOCKET:-/tmp/fc-1.socket}"
TAP_DEV="${TAP_DEV:-tap0}"
GUEST_IP="${GUEST_IP:-172.16.0.2}"
KEY="$(ls "$HOME"/fc-lab1/ubuntu-*.id_rsa 2>/dev/null | grep -v pub | tail -1)"

echo "=== 1. Host: find the VMM process ==="
pgrep -a firecracker || echo "no firecracker running (boot one first)"
FC_PID=$(pgrep -o firecracker || true)
[ -n "$FC_PID" ] && echo "FC_PID=$FC_PID"

echo "=== 2. Boundary inspection (see inspect-boundary.sh) ==="
bash "$(dirname "$0")/inspect-boundary.sh"

echo "=== 3. Guest probe (see guest-probe.sh) ==="
[ -n "${KEY:-}" ] && bash "$(dirname "$0")/guest-probe.sh" || echo "no SSH key; boot Lab 1 first"

echo "=== 4. Break things (see break-it.sh, interactive) ==="
echo "Run: bash $(dirname "$0")/break-it.sh"

echo "=== Cleanup ==="
echo "ssh -i $KEY root@$GUEST_IP reboot   # graceful shutdown"
echo "sudo ip link del $TAP_DEV 2>/dev/null || true; sudo rm -f $API_SOCKET"
