#!/usr/bin/env bash
# Verify Lab 3 matrix from the host.
set -u
GUEST_IP="${GUEST_IP:-172.16.0.2}"
TAP_DEV="${TAP_DEV:-tap0}"
KEY="$(ls "$HOME"/fc-lab1/ubuntu-*.id_rsa 2>/dev/null | grep -v pub | tail -1)"
SSH="ssh -i ${KEY:-/dev/null} -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$GUEST_IP"
pass=0; fail=0
chk() { if eval "$2"; then echo "PASS: $1"; pass=$((pass+1)); else echo "FAIL: $1"; fail=$((fail+1)); fi; }
chk "tap exists" "ip link show $TAP_DEV >/dev/null 2>&1"
chk "forwarding on" "[ \"\$(cat /proc/sys/net/ipv4/ip_forward)\" = 1 ]"
chk "NAT rule present" "sudo iptables -t nat -C POSTROUTING -o \"\$(ip -j route list default | jq -r '.[0].dev')\" -j MASQUERADE 2>/dev/null"
chk "host->guest ping" "ping -c2 -W2 $GUEST_IP >/dev/null 2>&1"
if [ -n "${KEY:-}" ]; then
  chk "guest->host ping" "$SSH 'ping -c2 -W2 172.16.0.1' >/dev/null 2>&1"
  chk "guest->internet ping" "$SSH 'ping -c2 -W2 8.8.8.8' >/dev/null 2>&1"
  chk "guest->internet DNS+curl" "$SSH 'curl -s -m 8 -o /dev/null -w %{http_code} https://example.com | grep -q 200' >/dev/null 2>&1"
else echo "SKIP: guest checks (no SSH key)"; fi
echo "--- ARP ---"; ip neigh show dev "$TAP_DEV" || true
echo "--- TAP counters ---"; ip -s link show "$TAP_DEV" | head -8 || true
echo "Result: $pass pass, $fail fail"
[ "$fail" -eq 0 ]
