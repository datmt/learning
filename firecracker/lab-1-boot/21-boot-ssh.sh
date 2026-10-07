#!/usr/bin/env bash
# Boot microVM with TAP + SSH (Lab 1 + Lab 3 combined minimal).
# Usage: bash 21-boot-ssh.sh
set -euo pipefail
WORKDIR="${WORKDIR:-$HOME/fc-lab1}"
cd "$WORKDIR"
API_SOCKET="/tmp/fc-1.socket"
LOGFILE="./fc-1.log"
TAP_DEV="${TAP_DEV:-tap0}"
TAP_IP="172.16.0.1"
MASK_SHORT="/30"
FC_MAC="06:00:AC:10:00:02"
GUEST_IP="172.16.0.2"
sudo rm -f "$API_SOCKET"

KERNEL="./$(ls vmlinux-* | tail -1)"
ROOTFS="./$(ls ubuntu-*.ext4 | tail -1)"
KEY="./$(ls ubuntu-*.id_rsa | grep -v pub | tail -1)"
BOOT_ARGS="console=ttyS0 reboot=k panic=1"
[ "$(uname -m)" = "aarch64" ] && BOOT_ARGS="keep_bootcon ${BOOT_ARGS}"

sudo ip link del "$TAP_DEV" 2>/dev/null || true
sudo ip tuntap add dev "$TAP_DEV" mode tap
sudo ip addr add "${TAP_IP}${MASK_SHORT}" dev "$TAP_DEV"
sudo ip link set dev "$TAP_DEV" up
sudo sh -c "echo 1 > /proc/sys/net/ipv4/ip_forward"
sudo iptables -P FORWARD ACCEPT
HOST_IFACE=$(ip -j route list default | jq -r '.[0].dev')
sudo iptables -t nat -C POSTROUTING -o "$HOST_IFACE" -j MASQUERADE 2>/dev/null \
  || sudo iptables -t nat -A POSTROUTING -o "$HOST_IFACE" -j MASQUERADE

sudo ./firecracker --api-sock "$API_SOCKET" --enable-pci &
sleep 1
sudo curl -s -X PUT --unix-socket "$API_SOCKET" --data "{\"log_path\": \"${LOGFILE}\", \"level\": \"Debug\", \"show_level\": true, \"show_log_origin\": true}" "http://localhost/logger"
sudo curl -s -X PUT --unix-socket "$API_SOCKET" --data "{\"kernel_image_path\": \"${KERNEL}\", \"boot_args\": \"${BOOT_ARGS}\"}" "http://localhost/boot-source"
sudo curl -s -X PUT --unix-socket "$API_SOCKET" --data '{"mem_size_mib": 256, "vcpu_count": 1}' "http://localhost/machine-config"
sudo curl -s -X PUT --unix-socket "$API_SOCKET" --data "{\"drive_id\": \"rootfs\", \"path_on_host\": \"${ROOTFS}\", \"is_root_device\": true, \"is_read_only\": false}" "http://localhost/drives/rootfs"
sudo curl -s -X PUT --unix-socket "$API_SOCKET" --data "{\"iface_id\": \"net1\", \"guest_mac\": \"${FC_MAC}\", \"host_dev_name\": \"${TAP_DEV}\"}" "http://localhost/network-interfaces/net1"
sleep 0.2
sudo curl -s -X PUT --unix-socket "$API_SOCKET" --data '{"action_type": "InstanceStart"}' "http://localhost/actions"
sleep 2
ssh -i "$KEY" -o StrictHostKeyChecking=no root@"$GUEST_IP" "ip route add default via 172.16.0.1 dev eth0; echo nameserver 8.8.8.8 > /etc/resolv.conf; echo nameserver 1.1.1.1 >> /etc/resolv.conf" || true
echo "SSH in with: ssh -i $KEY root@$GUEST_IP"
ssh -i "$KEY" -o StrictHostKeyChecking=no root@"$GUEST_IP"
