#!/usr/bin/env bash
# Lab 1 runbook — read top to bottom, copy-paste block by block.
# Host: Arch x86_64 (also works on Ubuntu/aarch64).
set -u
ARCH="$(uname -m)"
FC_VERSION="${FC_VERSION:-v1.16.1}"
WORKDIR="${WORKDIR:-$HOME/fc-lab1}"
mkdir -p "$WORKDIR" && cd "$WORKDIR"

echo "=== 1. Prereqs ==="
# Arch:
# sudo pacman -S --needed curl wget jq iproute2 iptables e2fsprogs squashfs-tools openssh --noconfirm
# Ubuntu:
# sudo apt install -y curl wget jq iproute2 iptables e2fsprogs squashfs-tools openssh-client
[ -r /dev/kvm ] && [ -w /dev/kvm ] && echo "KVM OK" || echo "KVM FAIL: sudo chmod 666 /dev/kvm"
lsmod | grep kvm || true

echo "=== 2. Install firecracker (see 00-install.sh) ==="
bash "$(dirname "$0")/00-install.sh"

echo "=== 3. Fetch kernel + rootfs (see 10-fetch-assets.sh) ==="
bash "$(dirname "$0")/10-fetch-assets.sh"

echo "=== 4a. Boot serial-console only (simplest) ==="
echo "Run: bash $(dirname "$0")/20-boot-serial.sh"
echo "Login root/root, then type 'reboot' to shut down."

echo "=== 4b. Boot with SSH/TAP (needs Lab 3 networking) ==="
echo "Run: bash $(dirname "$0")/21-boot-ssh.sh"
echo "Then: ssh -i ubuntu-*.id_rsa root@172.16.0.2"
