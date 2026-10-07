#!/usr/bin/env bash
# Read-only: where Docker keeps per-container mount metadata + enter its mount ns.
set -euo pipefail
NAME="${1:-mntdemo}"
docker inspect "$NAME" >/dev/null 2>&1 || { echo "Run ./demo.sh first."; exit 1; }
HOST_PID="$(docker inspect "$NAME" --format '{{.State.Pid}}')"
echo "==> Mounts (docker inspect):"
docker inspect "$NAME" --format '{{json .Mounts}}' | python3 -m json.tool
echo; echo "==> Identity file paths on host:"
docker inspect "$NAME" --format 'HostnamePath={{.HostnamePath}} HostsPath={{.HostsPath}} ResolvPath={{.ResolvConfPath}}'
echo; echo "==> Container mount table (no sudo: via docker exec):"
docker exec "$NAME" cat /proc/mounts | head -20
echo; echo "==> mnt namespace inodes (host vs container, no sudo):"
echo -n "host shell: "; readlink /proc/self/ns/mnt
echo -n "container : "; docker exec "$NAME" readlink /proc/1/ns/mnt
echo; echo "==> (Optional, needs sudo) same via nsenter: sudo nsenter -t $HOST_PID -m cat /proc/mounts | head -20"
