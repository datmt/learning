#!/usr/bin/env bash
# OOM demo: grow a bash string exponentially inside a 50MB container.
# No python/apt needed — pure bash. The kernel OOM-kills the container.
# Safe: only the throwaway container dies.
set -euo pipefail
IMAGE="${IMAGE:-ubuntu:22.04}"
echo "==> Running a 50MB container that eats memory exponentially (bash string doubling)..."
set +e
docker run --rm --memory=50m --name oomdemo "$IMAGE" \
  bash -c 'x=a; while true; do x="$x$x"; done'
RC=$?
set -e
echo "==> exit code: $RC (137 = SIGKILL/OOM is typical)"
echo "Tip: rerun WITHOUT --rm and inspect .State.OOMKilled:"
echo "  docker run --memory=50m --name oomkeep $IMAGE bash -c 'x=a; while true; do x=\"\$x\$x\"; done'"
echo "  docker inspect oomkeep --format 'OOMKilled={{.State.OOMKilled}}'; docker rm oomkeep"
