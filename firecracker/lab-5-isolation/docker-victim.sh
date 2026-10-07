#!/usr/bin/env bash
# Start two docker containers for comparison. Safe: sleep infinity, no priv.
set -u
if ! command -v docker >/dev/null; then echo "docker not installed; skipping"; exit 0; fi
sudo docker run -d --rm --name tenant-a --hostname tenant-a alpine sleep 3600 2>/dev/null || \
  sudo docker start tenant-a 2>/dev/null || true
sudo docker run -d --rm --name tenant-b --hostname tenant-b alpine sleep 3600 2>/dev/null || \
  sudo docker start tenant-b 2>/dev/null || true
sudo docker ps --filter name=tenant
echo "Try: sudo docker exec tenant-a uname -r   vs host uname -r"
