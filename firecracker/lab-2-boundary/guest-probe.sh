#!/usr/bin/env bash
# Guest-side probe via SSH. Proves guest has its own kernel/PID1/net.
set -u
GUEST_IP="${GUEST_IP:-172.16.0.2}"
KEY="$(ls "$HOME"/fc-lab1/ubuntu-*.id_rsa 2>/dev/null | grep -v pub | tail -1)"
[ -z "$KEY" ] && { echo "No SSH key. Boot Lab 1 first."; exit 1; }
SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$GUEST_IP"
echo "== Host kernel =="; uname -a
echo "== Guest kernel =="; $SSH "uname -a"
echo "== Guest PID1 =="; $SSH "ps -p 1 -o pid,comm,args"
echo "== Guest can NOT see host processes (look for firecracker: should be absent) =="
$SSH "ps aux | head -20; echo ---; ps aux | grep -c firecracker || echo '0 (good: host VMM invisible)'"
echo "== Guest mounts =="; $SSH "mount | grep -E ' / ' ; ls /"
echo "== Guest net (own stack; eth0 via virtio) =="; $SSH "ip addr; ip route"
echo "== Guest CPU (virtualized) =="; $SSH "grep -m1 'model name' /proc/cpuinfo; dmesg | grep -im1 -E 'kvm|virtio|Booting Linux'"
echo "== Host TAP counters move when guest pings =="
ping -c2 "$GUEST_IP" >/dev/null 2>&1 || true
ip -s link show "${TAP_DEV:-tap0}" | head -10 || true
