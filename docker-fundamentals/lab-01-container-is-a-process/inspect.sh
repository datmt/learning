#!/usr/bin/env bash
# Lab 1 inspect: deeper host-side forensics. Read-only (no changes).
set -euo pipefail
NAME="${1:-demo}"

if ! docker inspect "$NAME" >/dev/null 2>&1; then
  echo "Container '$NAME' not running. Run ./demo.sh first."; exit 1
fi

HOST_PID="$(docker inspect "$NAME" --format '{{.State.Pid}}')"
echo "==> Host PID: $HOST_PID"
echo "==> Full inspect (PID, PPID, runtime, shim):"
docker inspect "$NAME" --format 'Pid={{.State.Pid}}  PPid={{.State.PPid}}  Runtime={{.HostConfig.Runtime}}  InitPID/host PID match check: /proc/{{.State.Pid}}/cmdline exists={{fileExists (print "/proc/" .State.Pid "/cmdline")}}' 2>/dev/null \
  || docker inspect "$NAME" --format 'Pid={{.State.Pid}} Runtime={{.HostConfig.Runtime}}'

echo; echo "==> Namespace inodes: host shell vs container init"
echo "(reading container side via 'docker exec' — no sudo needed):"
echo -n "host shell pid ns: "; readlink /proc/self/ns/pid
echo -n "container pid ns : "; docker exec "$NAME" readlink /proc/1/ns/pid
echo "==> Full ns list from inside the container:"
docker exec "$NAME" ls -l /proc/1/ns/

echo; echo "==> Compare with your own shell's namespaces (different inodes = isolated):"
ls -l /proc/self/ns/

echo; echo "==> cgroup membership:"
cat "/proc/$HOST_PID/cgroup"

echo; echo "==> Kernel says the exe is plain sleep:"
cat "/proc/$HOST_PID/cmdline" | tr '\0' ' '; echo
