#!/usr/bin/env bash
# Lab 4 inspect: browse overlay dirs from the host + container disk usage.
set -euo pipefail
NAME="${1:-fsdemo}"
docker inspect "$NAME" >/dev/null 2>&1 || { echo "Run ./demo.sh first."; exit 1; }
LOWER="$(docker inspect "$NAME" --format '{{.GraphDriver.Data.LowerDir}}')"
UPPER="$(docker inspect "$NAME" --format '{{.GraphDriver.Data.UpperDir}}')"
MERGED="$(docker inspect "$NAME" --format '{{.GraphDriver.Data.MergedDir}}')"
echo "LowerDir: $LOWER" | tr ':' '\n' | head -8
echo "UpperDir: $UPPER"
echo "MergedDir: $MERGED"
echo; echo "--- UpperDir contents (your container's writes):"
ls -R "$UPPER" 2>/dev/null | head -30 || sudo ls -R "$UPPER" 2>/dev/null | head -30 || echo "(sudo needed: sudo ls -R $UPPER)"
echo; echo "--- MergedDir == container's / (spot check):"
ls "$MERGED" 2>/dev/null | head || sudo ls "$MERGED" | head; echo "..."
echo -n "host merged/hello.txt vs container /hello.txt: "
cat "$MERGED/hello.txt" 2>/dev/null || sudo cat "$MERGED/hello.txt" 2>/dev/null || echo "(no hello.txt yet — run demo.sh write test)"
echo; echo "--- image disk usage:"; docker system df
