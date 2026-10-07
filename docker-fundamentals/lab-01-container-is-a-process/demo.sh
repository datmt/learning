#!/usr/bin/env bash
# Lab 1 demo: prove a container is just a host process.
# Safe: starts one `sleep` container, prints host vs container views, leaves it running
# for you to explore. Run ./cleanup.sh afterwards.
set -euo pipefail

NAME="${1:-demo}"
IMAGE="${IMAGE:-ubuntu:22.04}"

echo "==> Pulling $IMAGE (cached if present)..."
docker pull "$IMAGE" >/dev/null || true

echo "==> Starting container '$NAME'..."
docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --rm --name "$NAME" "$IMAGE" sleep 10000 >/dev/null

echo
echo "==> docker ps:"
docker ps --filter "name=$NAME" --format 'table {{.Names}}\t{{.Image}}\t{{.Status}}\t{{.Command}}'

echo
echo "==> docker top $NAME  (host's view of container processes):"
docker top "$NAME"

HOST_PID="$(docker inspect "$NAME" --format '{{.State.Pid}}')"
echo
echo "==> Host PID of container init: $HOST_PID"
echo "==> ps on the host (ordinary process!):"
ps -o pid,ppid,stat,comm,args -p "$HOST_PID"
echo "==> /proc/$HOST_PID/exe:"
ls -l "/proc/$HOST_PID/exe" 2>/dev/null || sudo ls -l "/proc/$HOST_PID/exe" 2>/dev/null || echo "(need sudo to read exe link of root-owned process)"
echo -n "==> /proc/$HOST_PID/cmdline: "; tr '\0' ' ' < "/proc/$HOST_PID/cmdline"; echo

echo
echo "==> Container's view (docker exec):"
docker exec "$NAME" cat /proc/1/cmdline | tr '\0' ' ' ; echo "   <-- /proc/1/cmdline inside == host PID $HOST_PID outside"
docker exec "$NAME" bash -c 'echo "my shell PID inside: $$"; ls -l /proc/1/exe 2>/dev/null || cat /proc/1/status | head -5'

echo
echo "==> Parent chain (systemd -> ... -> sleep):"
if command -v pstree >/dev/null 2>&1; then
  pstree -p -s "$HOST_PID" || true
else
  echo "(install psmisc for pstree; falling back to PPID walk)"
  p="$HOST_PID"
  for _ in 1 2 3 4 5 6; do
    [ -r "/proc/$p/status" ] || break
    name="$(awk '/^Name:/{print $2}' "/proc/$p/status")"
    ppid="$(awk '/^PPid:/{print $2}' "/proc/$p/status")"
    echo "  PID $p ($name) <- PPID $ppid"
    [ "$ppid" = "0" ] || [ "$ppid" = "1" ] && break
    p="$ppid"
  done
fi

echo
echo "Done. Try:"
echo "  docker exec -it $NAME bash"
echo "  ps -o pid,ppid,comm -p $HOST_PID   # from host"
echo "  ./inspect.sh                       # deeper dive"
echo "  ./cleanup.sh                       # tear down"
