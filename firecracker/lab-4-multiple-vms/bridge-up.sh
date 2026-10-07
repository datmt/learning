#!/usr/bin/env bash
# Create fc-br0 bridge + NAT. Up part; see bridge-down.sh for teardown.
set -euo pipefail
BR="${BR:-fc-br0}"
BR_IP="${BR_IP:-10.0.0.1/24}"
HOST_IFACE="${HOST_IFACE:-$(ip -j route list default | jq -r '.[0].dev')}"
sudo ip link add name "$BR" type bridge 2>/dev/null || true
sudo ip addr add "$BR_IP" dev "$BR" 2>/dev/null || true
sudo ip link set dev "$BR" up
sudo sh -c "echo 1 > /proc/sys/net/ipv4/ip_forward"
sudo iptables -P FORWARD ACCEPT
sudo iptables -t nat -C POSTROUTING -o "$HOST_IFACE" -j MASQUERADE 2>/dev/null \
  || sudo iptables -t nat -A POSTROUTING -o "$HOST_IFACE" -j MASQUERADE
ip addr show "$BR"
echo "Bridge $BR up ($BR_IP), egress $HOST_IFACE"
