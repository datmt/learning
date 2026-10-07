#!/usr/bin/env bash
# Lab 6 demo: memory+cpu limited container and its cgroup v2 files.
set -euo pipefail
NAME="${1:-limited}"
IMAGE="${IMAGE:-nginx:alpine}"
docker pull "$IMAGE" >/dev/null || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --rm --name "$NAME" --memory=100m --cpus=0.5 "$IMAGE" >/dev/null

CID="$(docker inspect "$NAME" --format '{{.Id}}')"
PID="$(docker inspect "$NAME" --format '{{.State.Pid}}')"
SCOPE="/sys/fs/cgroup/system.slice/docker-${CID}.scope"
echo "=== $NAME: CID=${CID:0:12} PID=$PID ==="
echo "--- /proc/PID/cgroup:"; cat "/proc/$PID/cgroup"
echo "--- scope dir: $SCOPE"
if [ -d "$SCOPE" ]; then
  echo "memory.max : $(cat "$SCOPE/memory.max")  (100m = 104857600)"
  echo "cpu.max    : $(cat "$SCOPE/cpu.max")  (0.5 cpu = quota 50000 / period 100000)"
  echo "pids.max   : $(cat "$SCOPE/pids.max")"
  echo "memory.current: $(cat "$SCOPE/memory.current") bytes live"
  echo "--- cpu.stat:"; cat "$SCOPE/cpu.stat"
else
  echo "(scope not found — CgroupDriver may differ; see ./inspect.sh fallback)"
fi
echo; echo "--- docker stats:"; docker stats "$NAME" --no-stream
echo; echo "Try: ./stress.sh (OOM demo)  ./inspect.sh (full walk)  ./cleanup.sh"
