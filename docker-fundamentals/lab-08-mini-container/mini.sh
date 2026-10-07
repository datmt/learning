#!/usr/bin/env bash
# mini.sh — a ~70-line teaching "container runtime".
# Usage: sudo ./mini.sh [--mem 50m] [--pids 64] [--hostname mini-ctr] [--rootfs DIR] CMD [ARGS...]
# Needs: sudo/root (mount + cgroup writes), unshare, overlay kernel module.
set -euo pipefail

MEM="100m"; PIDS="64"; HOSTNAME="mini-ctr"; ROOTFS=""; LOWER_AUTO="/tmp/mini-lower-ubuntu"
args=(); CMD=()
while [ $# -gt 0 ]; do
  case "$1" in
    --mem) MEM="$2"; shift 2;;
    --pids) PIDS="$2"; shift 2;;
    --hostname) HOSTNAME="$2"; shift 2;;
    --rootfs) ROOTFS="$2"; shift 2;;
    --help|-h) sed -n '2,4p' "$0"; echo "Defaults: --mem 100m --pids 64 --hostname mini-ctr"; exit 0;;
    --) shift; CMD=("$@"); break;;
    -*) echo "unknown flag $1"; exit 1;;
    *) CMD=("$@"); break;;
  esac
done
[ ${#CMD[@]} -gt 0 ] || { echo "usage: sudo ./mini.sh [--mem 50m] CMD..."; exit 1; }
[ "$(id -u)" = "0" ] || { echo "need sudo/root (mount + cgroup)."; exit 1; }

# --- 1. Lower layer: Ubuntu rootfs (reuse Docker image if available) ---
if [ -z "$ROOTFS" ]; then
  if [ ! -d "$LOWER_AUTO" ] || [ -z "$(ls -A "$LOWER_AUTO" 2>/dev/null)" ]; then
    echo "==> fetching Ubuntu rootfs via docker export..." >&2
    mkdir -p "$LOWER_AUTO"
    if command -v docker >/dev/null 2>&1; then
      docker pull ubuntu:22.04 >/dev/null
      docker create --name mini-lower-tmp ubuntu:22.04 >/dev/null
      docker export mini-lower-tmp | tar -x -C "$LOWER_AUTO"
      docker rm mini-lower-tmp >/dev/null
    else
      echo "No docker and no --rootfs given. Install docker or debootstrap and retry." >&2; exit 1
    fi
  fi
  ROOTFS="$LOWER_AUTO"
fi

# --- 2. Overlay dirs: lower=read-only image, upper=writes, merged=container / ---
WORK="$(mktemp -d /tmp/mini-ctr-XXXXXX)"
UPPER="$WORK/upper"; WORKD="$WORK/work"; MERGED="$WORK/merged"
mkdir -p "$UPPER" "$WORKD" "$MERGED"
mount -t overlay overlay -o "lowerdir=$ROOTFS,upperdir=$UPPER,workdir=$WORKD" "$MERGED"
mkdir -p "$MERGED/proc" "$MERGED/sys" "$MERGED/dev" "$MERGED/tmp"

# --- 3. Cgroup (v2): memory + pids caps, self-cleaning ---
CG="/sys/fs/cgroup/mini-$(basename "$WORK")"
mkdir -p "$CG"
echo "+memory +pids" > /sys/fs/cgroup/cgroup.subtree_control 2>/dev/null || true
# mem like 50m/100m -> bytes
to_bytes(){ num="${1%[mMgGkK]}"; suf="${1#$num}"; case "$suf" in m|M) echo $((num*1024*1024));; g|G) echo $((num*1024*1024*1024));; k|K) echo $((num*1024));; *) echo "$1";; esac; }
echo "$(to_bytes "$MEM")" > "$CG/memory.max"
echo "$PIDS" > "$CG/pids.max"

cleanup(){
  umount -l "$MERGED/proc" 2>/dev/null || true
  umount -l "$MERGED/sys" 2>/dev/null || true
  umount -l "$MERGED/dev" 2>/dev/null || true
  umount -l "$MERGED" 2>/dev/null || true
  rmdir "$CG" 2>/dev/null || true
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "==> mini-ctr: hostname=$HOSTNAME mem=$MEM pids=$PIDS upper=$UPPER" >&2
echo "==> lower=$ROOTFS (read-only, shared like an image)" >&2

# --- 4. Enter namespaces and exec CMD as PID 1 of its world ---
# NOTE: --map-root-user needs a user ns (-U implied on some util-linux); add -U explicitly:
exec unshare --pid --fork --mount --uts --net -U --map-root-user \
  bash -c '
    MERGED="$1"; HOSTNAME="$2"; shift 2
    mount --make-rprivate / 2>/dev/null || true
    mount -t proc proc "$MERGED/proc"
    mount --bind /sys "$MERGED/sys" 2>/dev/null || true
    mount --bind /dev "$MERGED/dev" 2>/dev/null || true
    hostname "$HOSTNAME"
    # join cgroup so limits apply to everything below:
    echo $$ > "'"$CG"'/cgroup.procs"
    exec chroot "$MERGED" "$@"
  ' _ "$MERGED" "$HOSTNAME" "${CMD[@]}"
