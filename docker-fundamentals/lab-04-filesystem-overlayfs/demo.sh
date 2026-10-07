#!/usr/bin/env bash
# Lab 4 demo: layers + overlay dirs + copy-up proof.
set -euo pipefail
NAME="${1:-fsdemo}"
IMAGE="${IMAGE:-ubuntu:22.04}"
docker pull "$IMAGE" >/dev/null || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --rm --name "$NAME" "$IMAGE" sleep 10000 >/dev/null

echo "=== docker history (image layers) ==="
docker history "$IMAGE" --format 'table {{.CreatedBy}}\t{{.Size}}' | head -12

echo; echo "=== storage driver ==="
docker info 2>/dev/null | grep -iE 'storage driver|backing filesystem' || true

echo; echo "=== GraphDriver.Data (overlay2 dirs) ==="
docker inspect "$NAME" --format '{{json .GraphDriver.Data}}' | python3 -m json.tool

echo; echo "=== write test: new file + modify lower file ==="
docker exec "$NAME" bash -c 'echo hello-overlay > /hello.txt; echo "#touched" >> /etc/hostname-test 2>/dev/null || echo data >> /etc/hosts; ls /hello.txt'
echo "--- docker diff $NAME (A=added C=changed):"
docker diff "$NAME" | head -20

UPPER="$(docker inspect "$NAME" --format '{{.GraphDriver.Data.UpperDir}}')"
echo; echo "--- UpperDir on host ($UPPER):"
ls -la "$UPPER" 2>/dev/null | head -15 || sudo ls -la "$UPPER" 2>/dev/null | head -15 || echo "(need sudo to list $UPPER — try: sudo ls $UPPER)"

echo; echo "=== sharing test: second container has separate upper, same lowers ==="
docker run -d --rm --name "${NAME}B" "$IMAGE" sleep 10000 >/dev/null
docker exec "$NAME" touch /only-in-first
docker exec "${NAME}B" ls /only-in-first 2>&1 >/dev/null && echo "LEAKED (bad)" || echo "not visible in second container ✔ (separate upperdirs)"
docker rm -f "${NAME}B" >/dev/null
echo; echo "Cleanup: ./cleanup.sh"
