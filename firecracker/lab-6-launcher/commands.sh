#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
echo "=== API-socket mode ==="
echo "sudo python3 launcher.py boot --kernel ~/fc-lab1/vmlinux-* --rootfs ~/fc-lab1/ubuntu-*.ext4 --socket /tmp/fc-demo.socket --tap tap-demo"
echo "=== config-file mode ==="
echo "cp vm.json.example /tmp/vm.json  # edit paths, then:"
echo "sudo ./firecracker --api-sock /tmp/fc-cfg.socket --config-file /tmp/vm.json"
echo "=== full demo ==="
bash ./demo.sh
