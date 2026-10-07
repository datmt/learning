#!/usr/bin/env bash
# Boot microVM with serial console only (no network). Blocking: shows guest logs.
# Usage: bash 20-boot-serial.sh
# Login: root / root. Type `reboot` to shut down.
set -euo pipefail
WORKDIR="${WORKDIR:-$HOME/fc-lab1}"
cd "$WORKDIR"
API_SOCKET="/tmp/fc-1.socket"
LOGFILE="./fc-1.log"
sudo rm -f "$API_SOCKET"

KERNEL="./$(ls vmlinux-* | tail -1)"
ROOTFS="./$(ls ubuntu-*.ext4 | tail -1)"
BOOT_ARGS="console=ttyS0 reboot=k panic=1"
[ "$(uname -m)" = "aarch64" ] && BOOT_ARGS="keep_bootcon ${BOOT_ARGS}"
echo "Kernel: $KERNEL"
echo "Rootfs: $ROOTFS"

# Start firecracker in background; logs stream to this terminal via --serial-file? No:
# simplest is foreground in another terminal. Here we background it and tail the log.
sudo ./firecracker --api-sock "$API_SOCKET" --enable-pci &
FC_PID=$!
sleep 1

sudo curl -s -X PUT --unix-socket "$API_SOCKET" --data "{\"log_path\": \"${LOGFILE}\", \"level\": \"Debug\", \"show_level\": true, \"show_log_origin\": true}" "http://localhost/logger"
sudo curl -s -X PUT --unix-socket "$API_SOCKET" --data "{\"kernel_image_path\": \"${KERNEL}\", \"boot_args\": \"${BOOT_ARGS}\"}" "http://localhost/boot-source"
sudo curl -s -X PUT --unix-socket "$API_SOCKET" --data "{\"mem_size_mib\": 256, \"vcpu_count\": 1, \"smt\": false}" "http://localhost/machine-config"
sudo curl -s -X PUT --unix-socket "$API_SOCKET" --data "{\"drive_id\": \"rootfs\", \"path_on_host\": \"${ROOTFS}\", \"is_root_device\": true, \"is_read_only\": false}" "http://localhost/drives/rootfs"
sleep 0.1
sudo curl -s -X PUT --unix-socket "$API_SOCKET" --data '{"action_type": "InstanceStart"}' "http://localhost/actions"
echo "Booted. Firecracker PID on host: $FC_PID"
echo "Serial console is the firecracker stdout — if backgrounded, attach with:"
echo "  sudo tail -f $LOGFILE"
echo "Version check:"
sudo curl -s --unix-socket "$API_SOCKET" "http://localhost/version" || true
echo
wait $FC_PID  # exits when guest `reboot`s
