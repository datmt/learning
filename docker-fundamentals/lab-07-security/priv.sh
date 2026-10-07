#!/usr/bin/env bash
# DANGEROUS contrast: default vs --privileged. LAB VM ONLY. Read-only-ish but removes guardrails.
set -euo pipefail
IMAGE="${IMAGE:-ubuntu:22.04}"
echo "WARNING: --privileged gives near-host-root. Isolated lab machine only. Ctrl-C to abort."
sleep 3
echo; echo "=== CapEff ==="
echo -n "default    : "; docker run --rm "$IMAGE" grep CapEff /proc/self/status
echo -n "privileged : "; docker run --rm --privileged "$IMAGE" grep CapEff /proc/self/status
echo; echo "=== devices (privileged sees host /dev) ==="
echo "--- default:"; docker run --rm "$IMAGE" ls /dev | tr '\n' ' '; echo
echo "--- privileged:"; docker run --rm --privileged "$IMAGE" ls /dev | tr '\n' ' '; echo
echo; echo "=== can privileged mount? ==="
docker run --rm --privileged "$IMAGE" sh -c 'mkdir -p /mnt/t && mount -t tmpfs tmpfs /mnt/t && mount | grep /mnt/t && umount /mnt/t && echo "mount works under --privileged (NEVER in prod)"'
echo; echo "Moral: ship --cap-drop=ALL + minimal adds, keep seccomp, never --privileged in prod."
