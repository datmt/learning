#!/usr/bin/env bash
# Full cgroup walk with systemd-driver fallback search. Read-only.
set -euo pipefail
NAME="${1:-limited}"
docker inspect "$NAME" >/dev/null 2>&1 || { echo "Run ./demo.sh first."; exit 1; }
CID="$(docker inspect "$NAME" --format '{{.Id}}')"
PID="$(docker inspect "$NAME" --format '{{.State.Pid}}')"
echo "==> HostConfig limits:"
docker inspect "$NAME" --format 'Memory={{.HostConfig.Memory}} NanoCpus={{.HostConfig.NanoCpus}} PidsLimit={{.HostConfig.PidsLimit}}'
echo; echo "==> cgroup path from /proc:"
cat "/proc/$PID/cgroup"
echo; echo "==> locating scope dir..."
SCOPE="$(grep -o '/system.slice/docker-.*\.scope' "/proc/$PID/cgroup" | head -1)"
[ -n "${SCOPE:-}" ] && SCOPE="/sys/fs/cgroup$SCOPE"
if [ -z "${SCOPE:-}" ] || [ ! -d "$SCOPE" ]; then
  echo "(systemd path not found, searching by container ID...)"
  SCOPE="$(grep -rl "$CID" /sys/fs/cgroup/system.slice/ 2>/dev/null | head -1 | xargs dirname 2>/dev/null || true)"
fi
echo "SCOPE=$SCOPE"
[ -n "${SCOPE:-}" ] && [ -d "$SCOPE" ] || { echo "Could not locate cgroup dir (driver?). CgroupVersion: $(docker info 2>/dev/null | grep -i cgroup)"; exit 1; }
for f in memory.max memory.current memory.events cpu.max cpu.stat pids.max pids.current cgroup.procs cgroup.controllers; do
  [ -f "$SCOPE/$f" ] && { echo "--- $f:"; cat "$SCOPE/$f"; }
done
echo; echo "--- inside view (cgroup ns):"
docker exec "$NAME" cat /sys/fs/cgroup/memory.current 2>/dev/null || docker exec "$NAME" cat /sys/fs/cgroup/memory/memory.usage_in_bytes 2>/dev/null || echo "(no cgroupfs inside image)"
