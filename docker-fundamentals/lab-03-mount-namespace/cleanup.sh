#!/usr/bin/env bash
set -euo pipefail
NAME="${1:-mntdemo}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
rm -rf /tmp/mntdemo-lab /tmp/host-only-lab3.txt
echo "Cleaned up '$NAME'."
