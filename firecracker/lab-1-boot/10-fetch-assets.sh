#!/usr/bin/env bash
# Fetch guest kernel + Ubuntu rootfs from Firecracker CI (S3).
# Produces: vmlinux-<ver>, ubuntu-<ver>.ext4, ubuntu-<ver>.id_rsa
set -euo pipefail
ARCH="$(uname -m)"
WORKDIR="${WORKDIR:-$HOME/fc-lab1}"
mkdir -p "$WORKDIR" && cd "$WORKDIR"
S3="https://s3.amazonaws.com/spec.ccfc.min"
CURL_OPTS=(-fsSL --retry 8 --retry-delay 3 --retry-all-errors --connect-timeout 15 --max-time 60)

echo "Listing CI artifacts for ${ARCH}..."
CI_PREFIX=$(curl "${CURL_OPTS[@]}" "$S3?list-type=2&prefix=firecracker-ci/&delimiter=/" \
  | grep -oP '(?<=<Prefix>)firecracker-ci/[0-9]{8}-[^/]+/(?=</Prefix>)' \
  | sort | tail -1)
[ -n "${CI_PREFIX:-}" ] || { echo "ERROR: empty CI prefix (S3 listing failed, re-run)"; exit 1; }
echo "CI prefix: ${CI_PREFIX}"

KKEY=$(curl "${CURL_OPTS[@]}" "$S3?list-type=2&prefix=${CI_PREFIX}${ARCH}/vmlinux-" \
  | grep -oP "(?<=<Key>)(${CI_PREFIX}${ARCH}/vmlinux-[0-9]+\\.[0-9]+\\.[0-9]{1,3})(?=</Key>)" \
  | sort -V | tail -1)
[ -n "${KKEY:-}" ] || { echo "ERROR: empty kernel key (S3 listing failed, re-run)"; exit 1; }
echo "Kernel key: ${KKEY}"
wget --tries=10 --retry-connrefused --waitretry=3 --timeout=30 -N "$S3/${KKEY}"

UKEY=$(curl "${CURL_OPTS[@]}" "$S3?list-type=2&prefix=${CI_PREFIX}${ARCH}/ubuntu-" \
  | grep -oP "(?<=<Key>)(${CI_PREFIX}${ARCH}/ubuntu-[0-9]+\\.[0-9]+\\.squashfs)(?=</Key>)" \
  | sort -V | tail -1)
[ -n "${UKEY:-}" ] || { echo "ERROR: empty ubuntu key (S3 listing failed, re-run)"; exit 1; }
UVER=$(basename "$UKEY" .squashfs | grep -oE '[0-9]+\.[0-9]+')
echo "Ubuntu key: ${UKEY} (version ${UVER})"
# Resume-capable download for the ~108MB squashfs (wget -O restarts from 0 on retry)
curl -fSL --retry 8 --retry-delay 3 --retry-all-errors --connect-timeout 15 -C - \
  "$S3/$UKEY" -o "ubuntu-${UVER}.squashfs.upstream"

if [ ! -f "ubuntu-${UVER}.id_rsa" ]; then
  rm -rf squashfs-root
  unsquashfs "ubuntu-${UVER}.squashfs.upstream"
  ssh-keygen -f id_rsa -N "" -C "fc-lab1"
  sudo mkdir -p squashfs-root/root/.ssh
  sudo cp -v id_rsa.pub squashfs-root/root/.ssh/authorized_keys
  mv -v id_rsa "./ubuntu-${UVER}.id_rsa"
  mv -v id_rsa.pub "./ubuntu-${UVER}.id_rsa.pub"
  sudo chown -R root:root squashfs-root
  truncate -s 1G "ubuntu-${UVER}.ext4"
  sudo mkfs.ext4 -d squashfs-root -F "ubuntu-${UVER}.ext4"
fi

echo
KERNEL=$(ls vmlinux-* | tail -1)
ROOTFS=$(ls ubuntu-*.ext4 | tail -1)
KEY=$(ls ubuntu-*.id_rsa | grep -v pub | tail -1)
[ -f "$KERNEL" ] && echo "Kernel: $KERNEL" || { echo "ERROR: no kernel"; exit 1; }
e2fsck -fn "$ROOTFS" >/dev/null && echo "Rootfs: $ROOTFS" || { echo "ERROR: bad ext4"; exit 1; }
[ -f "$KEY" ] && echo "SSH key: $KEY"
