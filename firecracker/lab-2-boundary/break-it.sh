#!/usr/bin/env bash
# Chaos menu: deliberately break things, observe blast radius.
set -u
GUEST_IP="${GUEST_IP:-172.16.0.2}"
TAP_DEV="${TAP_DEV:-tap0}"
KEY="$(ls "$HOME"/fc-lab1/ubuntu-*.id_rsa 2>/dev/null | grep -v pub | tail -1)"
SSH="ssh -i ${KEY:-/dev/null} -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$GUEST_IP"
PS3="Break what? "
select opt in "STOP-VMM-freezes-guest" "delete-TAP-kills-net-only" "fill-guest-disk" "guest-reboot" "quit"; do
  case $REPLY in
    1) FC_PID=$(pgrep -o firecracker); echo "STOP $FC_PID for 5s (guest clock freezes, host fine)..."
       sudo kill -STOP "$FC_PID"; sleep 5; sudo kill -CONT "$FC_PID"; echo "resumed.";;
    2) echo "Deleting $TAP_DEV (guest lives, net dies)..."; sudo ip link del "$TAP_DEV" || true;
       $SSH "ip link" || echo "guest unreachable (expected)";;
    3) echo "Filling guest /tmp (host disk barely moves: rootfs is a file)..."
       $SSH "dd if=/dev/zero of=/tmp/fill bs=1M count=100; df -h /; rm -f /tmp/fill" || echo "guest unreachable";;
    4) echo "Graceful shutdown via guest reboot..."; $SSH "reboot" || true;;
    5) break;;
    *) echo "pick 1-5";;
  esac
done
