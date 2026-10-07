#!/usr/bin/env bash
# Host-side TAP + forwarding + NAT. Usage: host-net.sh [up|down]
set -euo pipefail
TAP_DEV="${TAP_DEV:-tap0}"
TAP_IP="${TAP_IP:-172.16.0.1}"
MASK_SHORT="${MASK_SHORT:-/30}"
HOST_IFACE="${HOST_IFACE:-$(ip -j route list default | jq -r '.[0].dev')}"

up() {
  sudo ip link del "$TAP_DEV" 2>/dev/null || true
  sudo ip tuntap add dev "$TAP_DEV" mode tap
  sudo ip addr add "${TAP_IP}${MASK_SHORT}" dev "$TAP_DEV"
  sudo ip link set dev "$TAP_DEV" up
  sudo sh -c "echo 1 > /proc/sys/net/ipv4/ip_forward"
  sudo iptables -P FORWARD ACCEPT
  echo "Egress iface: $HOST_IFACE"
  sudo iptables -t nat -C POSTROUTING -o "$HOST_IFACE" -j MASQUERADE 2>/dev/null \
    || sudo iptables -t nat -A POSTROUTING -o "$HOST_IFACE" -j MASQUERADE
  ip addr show "$TAP_DEV"
  echo "UP: $TAP_DEV $TAP_IP$MASK_SHORT via $HOST_IFACE"
}
down() {
  sudo ip link del "$TAP_DEV" 2>/dev/null || true
  echo "DOWN: $TAP_DEV removed (NAT rule left in place; harmless)"
}
case "${1:-up}" in up) up;; down) down;; *) echo "usage: $0 [up|down]"; exit 1;; esac
