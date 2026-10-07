#!/usr/bin/env bash
# Lab 7 demo: default caps are a small subset of host-root caps; cap-add/drop in action.
# Uses a tiny tools image (lab7-tools:local) with iproute2 + iputils-ping + libcap2-bin.
set -euo pipefail
BASE="${IMAGE:-ubuntu:22.04}"
TOOLS="lab7-tools:local"
cd "$(dirname "$0")"
if ! docker image inspect "$TOOLS" >/dev/null 2>&1; then
  echo "==> Building $TOOLS (one-time, ~1 min: apt install iproute2 iputils-ping libcap2-bin)..."
  docker pull "$BASE" >/dev/null || true
  docker build -f Dockerfile.tools -t "$TOOLS" . 
fi
IMG="$TOOLS"

echo "=== whoami (both say root — powers differ) ==="
echo -n "host init (PID 1, real root): "; awk '/CapEff/{print $2}' /proc/1/status
echo -n "default container          : "; docker run --rm "$IMG" awk '/CapEff/{print $2}' /proc/self/status
echo "(your own shell has almost none: $(awk '/CapEff/{print $2}' /proc/self/status) — you are an unprivileged user)"
echo -n "privileged container       : "; docker run --rm --privileged "$IMG" awk '/CapEff/{print $2}' /proc/self/status 2>/dev/null || echo "(skipped)"

echo; echo "=== decoded default-container caps ==="
EFF="$(docker run --rm "$IMG" awk '/CapEff/{print $2}' /proc/self/status)"
docker run --rm "$IMG" capsh --decode="$EFF"

echo; echo "=== missing NET_ADMIN proof ==="
if docker run --rm "$IMG" ip link add dummy0 type dummy 2>&1; then echo "(unexpectedly allowed)"; else echo "denied by default ✔ (needs CAP_NET_ADMIN)"; fi
echo "--- with --cap-add=NET_ADMIN:"
docker run --rm --cap-add=NET_ADMIN "$IMG" sh -c 'ip link add dummy0 type dummy && ip link show dummy0 | head -2 && ip link del dummy0 && echo works ✔'

echo; echo "=== least privilege: --cap-drop=ALL breaks ping (needs NET_RAW) ==="
docker run --rm --cap-drop=ALL "$IMG" ping -c1 -W2 8.8.8.8 2>&1 | head -3 || true

echo; echo "=== seccomp ==="
docker info 2>/dev/null | grep -i -A1 seccomp || true
echo; echo "Next: ./priv.sh (privileged contrast — LAB VM ONLY)  ./inspect.sh (decode + checklist)"
