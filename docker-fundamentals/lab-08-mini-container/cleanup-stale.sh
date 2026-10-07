#!/usr/bin/env bash
# Remove stale mounts/cgroups from killed mini.sh runs. Safe to run anytime.
set -euo pipefail
for m in $(grep -o '/tmp/mini-ctr-[A-Za-z0-9]*[^ ]*' /proc/mounts 2>/dev/null | sort -u); do
  echo "unmounting $m"; sudo umount -l "$m" 2>/dev/null || true
done
for d in /tmp/mini-ctr-*; do [ -d "$d" ] && { echo "removing $d"; sudo rm -rf "$d"; }; done
for c in /sys/fs/cgroup/mini-mini-ctr-* /sys/fs/cgroup/mini-*; do [ -d "$c" ] && { echo "removing $c"; sudo rmdir "$c" 2>/dev/null || true; }; done
docker rm -f mini-lower-tmp >/dev/null 2>&1 || true
echo "Stale mini-container state cleaned."
