#!/usr/bin/env bash
# Boot one of A|B|C. Usage: boot-vm.sh <A|B|C>
set -euo pipefail
VM="${1:?usage: boot-vm.sh <A|B|C>}"
BR="${BR:-fc-br0}"
WORKDIR="${WORKDIR:-$HOME/fc-lab1}"
LOWER=$(echo "$VM" | tr '[:upper:]' '[:lower:]')
case "$VM" in
  A) TAP=tap-a; IP=10.0.0.2; MAC=06:00:AC:10:00:02;;
  B) TAP=tap-b; IP=10.0.0.3; MAC=06:00:AC:10:00:03;;
  C) TAP=tap-c; IP=10.0.0.4; MAC=06:00:AC:10:00:04;;
  *) echo "unknown VM $VM"; exit 1;;
esac
SOCK="/tmp/fc-${LOWER}.socket"
LOG="/tmp/fc-${LOWER}.log"
cd "$WORKDIR"
KERNEL="./$(ls vmlinux-* | tail -1)"
BASE_ROOTFS="./$(ls ubuntu-*.ext4 | tail -1)"
KEY="./$(ls ubuntu-*.id_rsa | grep -v pub | tail -1)"
VM_ROOTFS="/tmp/fc-rootfs-${LOWER}.ext4"
BOOT_ARGS="console=ttyS0 reboot=k panic=1"
[ "$(uname -m)" = "aarch64" ] && BOOT_ARGS="keep_bootcon ${BOOT_ARGS}"

cp -f "$BASE_ROOTFS" "$VM_ROOTFS"
sudo ip link del "$TAP" 2>/dev/null || true
sudo ip tuntap add dev "$TAP" mode tap
sudo ip link set dev "$TAP" up
sudo ip link set dev "$TAP" master "$BR"
sudo rm -f "$SOCK"
sudo ./firecracker --api-sock "$SOCK" --enable-pci &
sleep 1
sudo curl -s -X PUT --unix-socket "$SOCK" --data "{\"log_path\": \"${LOG}\", \"level\": \"Info\", \"show_level\": true, \"show_log_origin\": true}" "http://localhost/logger"
sudo curl -s -X PUT --unix-socket "$SOCK" --data "{\"kernel_image_path\": \"${KERNEL}\", \"boot_args\": \"${BOOT_ARGS}\"}" "http://localhost/boot-source"
sudo curl -s -X PUT --unix-socket "$SOCK" --data '{"mem_size_mib": 256, "vcpu_count": 1}' "http://localhost/machine-config"
sudo curl -s -X PUT --unix-socket "$SOCK" --data "{\"drive_id\": \"rootfs\", \"path_on_host\": \"${VM_ROOTFS}\", \"is_root_device\": true, \"is_read_only\": false}" "http://localhost/drives/rootfs"
sudo curl -s -X PUT --unix-socket "$SOCK" --data "{\"iface_id\": \"net1\", \"guest_mac\": \"${MAC}\", \"host_dev_name\": \"${TAP}\"}" "http://localhost/network-interfaces/net1"
sleep 0.2
sudo curl -s -X PUT --unix-socket "$SOCK" --data '{"action_type": "InstanceStart"}' "http://localhost/actions"
sleep 3
ssh -i "$KEY" -o StrictHostKeyChecking=no "root@${IP}" "ip addr add ${IP}/24 dev eth0 2>/dev/null; ip link set eth0 up; ip route add default via 10.0.0.1 dev eth0 2>/dev/null; echo nameserver 8.8.8.8 > /etc/resolv.conf" || echo "WARN: SSH config for VM-$VM failed; check $LOG"
echo "VM-$VM up: $IP via $TAP ($SOCK)"
