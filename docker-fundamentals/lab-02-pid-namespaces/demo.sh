#!/usr/bin/env bash
# Lab 2 demo: host PID vs container PID 1, namespace inodes.
set -euo pipefail
NAME="${1:-piddemo}"
IMAGE="${IMAGE:-ubuntu:22.04}"
docker pull "$IMAGE" >/dev/null || true
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --rm --name "$NAME" "$IMAGE" sleep 10000 >/dev/null

HOST_PID="$(docker inspect "$NAME" --format '{{.State.Pid}}')"
echo "==> Host PID: $HOST_PID"
echo "==> Host view:"; ps -o pid,ppid,stat,args -p "$HOST_PID"
echo; echo "==> Container view:"
docker exec "$NAME" bash -c 'echo "shell PID inside: $$"; echo -n "/proc/1/cmdline: "; tr "\0" " " < /proc/1/cmdline; echo; echo "--- /proc (only container procs):"; ls /proc | grep -E "^[0-9]+$" | tr "\n" " "; echo'
echo; echo "==> PID namespace inodes (different = isolated; read container-side, no sudo needed):"
echo -n "host shell: "; readlink /proc/self/ns/pid
echo -n "container : "; docker exec "$NAME" readlink /proc/1/ns/pid
echo "(host /proc/$HOST_PID/ns is root-only; 'docker exec ... readlink /proc/1/ns/pid' shows the same value)"
echo; echo "Try: sudo nsenter -t $HOST_PID -p ps aux   (host shell, container PID view)"
echo "Cleanup: ./cleanup.sh"
