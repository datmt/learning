#!/usr/bin/env bash
# Read-only forensics: nsenter into the container's PID namespace from the host.
set -euo pipefail
NAME="${1:-piddemo}"
docker inspect "$NAME" >/dev/null 2>&1 || { echo "Run ./demo.sh first."; exit 1; }
HOST_PID="$(docker inspect "$NAME" --format '{{.State.Pid}}')"
echo "==> Host PID $HOST_PID — container PID view WITHOUT sudo (docker exec):"
docker exec "$NAME" sh -c 'ls /proc | grep -E "^[0-9]+$" | tr "\n" " "; echo; cat /proc/1/cmdline | tr "\0" " "; echo'
echo; echo "==> All 7 namespace inodes of container init (via docker exec, no sudo):"
docker exec "$NAME" ls -l /proc/1/ns/
echo; echo "==> (Optional, needs sudo password) host-side nsenter for the full experience:"
echo "    sudo nsenter -t $HOST_PID -p -m ps aux"
