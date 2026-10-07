#!/usr/bin/env bash
# Fetch guest kernel + Ubuntu rootfs from Firecracker CI (S3).
# Produces: vmlinux-<ver>, ubuntu-<ver>.ext4, ubuntu-<ver>.id_rsa
set -euo pipefail
ARCH="$(uname -m)"
WORKDIR="${WORKDIR:-$HOME/fc-lab1}"
mkdir -p "$WORKDIR" && cd "$WORKDIR"
S3="https://s3.amazonaws.com/spec.ccfc.min"

echo "Listing CI artifacts for ${ARCH}..."
CI_PREFIX=$(curl -fsSL "$S3?list-type=2&prefix=firecracker-ci/&delimiter=/" \
  | grep -oP '(?<=<Prefix>)firecracker-ci/[0-9]{8}-[^/]+/(?=</Prefix>)' \
  | sort | tail -1)
echo "CI prefix: ${CI_PREFIX}"

KKEY=$(curl -fsSL "$S3?list-type=2&prefix=${CI_PREFIX}${ARCH}/vmlinux-" \
  | grep -oP "(?<=<Key>)(${CI_PREFIX}${ARCH}/vmlinux-[0-9]+\\.[0-9]+\\.[0-9]{1,3})(?=</Key>)" \
  | sort -V | tail -1)
echo "Kernel key: ${KKEY}"
wget -N "$S3/${KKEY}"

UKEY=$(curl -fsSL "$S3?list-type=2&prefix=${CI_PREFIX}${ARCH}/ubuntu-" \
  | grep -oP "(?<=<Key>)(${CI_PREFIX}${ARCH}/ubuntu-[0-9]+\\.[0-9]+\\.squashfs)(?=</Key>)" \
  | sort -V | tail -1)
UVER=$(basename "$UKEY" .squashfs | grep -oE '[0-9]+\.[0-9]+')
echo "Ubuntu key: ${UKEY} (version ${UVER})"
wget -O "ubuntu-${UVER}.squashfs.upstream" "$S3/$UKEY"

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
