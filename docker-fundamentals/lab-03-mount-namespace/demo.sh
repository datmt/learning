#!/usr/bin/env bash
# Lab 3 demo: host vs container mount/filesystem views.
set -euo pipefail
NAME="${1:-mntdemo}"
IMAGE="${IMAGE:-ubuntu:22.04}"
docker pull "$IMAGE" >/dev/null || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --rm --name "$NAME" "$IMAGE" sleep 10000 >/dev/null

echo "=== OS identity (filesystem, NOT kernel) ==="
echo "--- host:"; head -3 /etc/os-release
echo "--- container:"; docker exec "$NAME" bash -c 'head -3 /etc/os-release'

echo; echo "=== kernel (SAME both sides) ==="
echo "--- host:     $(uname -r)"
echo "--- container: $(docker exec "$NAME" uname -r)"

echo; echo "=== mount table sizes ==="
echo "host entries:      $(mount | wc -l)"
echo "container entries: $(docker exec "$NAME" mount | wc -l)"
echo "--- container mounts (first 15):"
docker exec "$NAME" mount | head -15

echo; echo "=== df -h (container) ==="
docker exec "$NAME" df -h | head -10

echo; echo "=== Docker-injected identity files ==="
echo -n "hostname: "; docker exec "$NAME" cat /etc/hostname
echo "--- /etc/hosts:"; docker exec "$NAME" cat /etc/hosts

echo; echo "=== Visibility test (/tmp/host-only.txt) ==="
echo secret > /tmp/host-only-lab3.txt
if docker exec "$NAME" cat /tmp/host-only-lab3.txt >/dev/null 2>&1; then echo "VISIBLE (unexpected!)"; else echo "NOT VISIBLE — mount isolation works ✔"; fi
echo "--- with -v it becomes visible (opt-in):"
docker run --rm -v /tmp:/tmp "$IMAGE" cat /tmp/host-only-lab3.txt
rm -f /tmp/host-only-lab3.txt

echo; echo "Cleanup: ./cleanup.sh"
