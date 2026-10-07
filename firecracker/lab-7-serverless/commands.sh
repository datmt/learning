#!/usr/bin/env bash
set -u
cd "$(dirname "$0")"
echo "=== start server (root needed) ==="
echo "sudo python3 server.py --port 8080 --max-vms 3 &"
echo "=== client ==="
bash ./client.sh || echo "(start server first)"
