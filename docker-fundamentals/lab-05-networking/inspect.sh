#!/usr/bin/env bash
# Lab 5 inspect: NAT rules, bridge members, cross-container ping. Read-only.
set -euo pipefail
NAME="${1:-web}"
docker inspect "$NAME" >/dev/null 2>&1 || { echo "Run ./demo.sh first."; exit 1; }
PID="$(docker inspect "$NAME" --format '{{.State.Pid}}')"
IP="$(docker inspect "$NAME" --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')"

echo "=== NAT / port-forward rules (needs sudo; skipped gracefully if no password) ==="
if sudo -n true 2>/dev/null; then
  sudo iptables -t nat -L DOCKER -n --line-numbers 2>/dev/null | head -25 || \
  sudo nft list ruleset 2>/dev/null | grep -A15 -i docker | head -40 || \
  sudo iptables -t nat -L -n 2>/dev/null | head -20
else
  echo "(sudo needs a password — run manually: sudo iptables -t nat -L DOCKER -n)"
fi

echo; echo "=== bridge members ==="
bridge link 2>/dev/null | head -10 || ip link show type bridge 2>/dev/null | head

echo; echo "=== container eth0 peer index (match with host veth; no sudo) ==="
docker exec "$NAME" ip -o link show eth0

echo; echo "=== cross-container ping over docker0 ==="
docker run -d --rm --name pingpeer nginx:alpine >/dev/null
sleep 1
PEER_IP="$(docker inspect pingpeer --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')"
echo "web($IP) -> pingpeer($PEER_IP):"
docker exec "$NAME" ping -c2 -W2 "$PEER_IP" | tail -3
docker rm -f pingpeer >/dev/null

echo; echo "=== isolation contrast ==="
echo "--- --network none:"; docker run --rm --network none ubuntu:22.04 ip -o addr 2>/dev/null | head -5 || docker run --rm --network none ubuntu:22.04 ip addr | head -8
