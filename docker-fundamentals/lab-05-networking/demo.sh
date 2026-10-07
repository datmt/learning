#!/usr/bin/env bash
# Lab 5 demo: bridge nginx + veth + NAT + curl test.
set -euo pipefail
NAME="${1:-web}"
IMAGE="${IMAGE:-nginx:alpine}"
# Pick a free host port (8080 is often taken): honour $PORT if set and free.
candidates="${PORT:-} 8080 8081 18080 18081"
PORT=""
for p in $candidates; do
  [ -z "$p" ] && continue
  if (command -v ss >/dev/null && ss -ltn 2>/dev/null | grep -q ":$p ") || docker ps --format '{{.Ports}}' | grep -q ":$p->"; then
    continue
  fi
  PORT="$p"; break
done
[ -n "$PORT" ] || { echo "no free port among candidates"; exit 1; }
echo "==> Using host port $PORT"
docker pull "$IMAGE" >/dev/null || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --rm --name "$NAME" -p "$PORT:80" "$IMAGE" >/dev/null

sleep 2
IP="$(docker inspect "$NAME" --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')"
PID="$(docker inspect "$NAME" --format '{{.State.Pid}}')"
echo "=== container $NAME: IP=$IP hostPID=$PID ==="
echo "--- inside: ip addr / ip route"
docker exec "$NAME" ip addr | grep -E '^[0-9]+:|inet ' | head -10
docker exec "$NAME" ip route
echo; echo "--- host: docker0 + veth peers"
ip addr show docker0 2>/dev/null | grep -E 'inet |mtu' | head -5 || echo "(no docker0?)"
ip -o link | grep -E 'veth|docker' | head -10
echo; echo "--- net ns inodes (isolated if different; no sudo)"
echo -n "host: "; readlink /proc/self/ns/net
echo -n "ctr : "; docker exec "$NAME" readlink /proc/1/ns/net
echo; echo "--- curl localhost:$PORT (through NAT into container:80)"
curl -s -o /dev/null -w 'HTTP %{http_code}\n' "localhost:$PORT"
echo; echo "Deeper: ./inspect.sh   Teardown: ./cleanup.sh"
