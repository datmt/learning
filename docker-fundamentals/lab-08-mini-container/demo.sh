#!/usr/bin/env bash
# Guided tour of mini.sh proofs. Needs sudo (mounts).
set -euo pipefail
cd "$(dirname "$0")"
echo "=== 1. PID 1 illusion ==="
sudo ./mini.sh /bin/bash -c 'echo "PID inside: $$ (want 1)"; ls /proc | grep -E "^[0-9]+$" | tr "\n" " "; echo'
echo; echo "=== 2. Hostname + mounts ==="
sudo ./mini.sh --hostname tiny /bin/bash -c 'hostname; echo ---; mount | head -6'
echo; echo "=== 3. Overlay writes stay in upperdir ==="
sudo ./mini.sh /bin/bash -c 'echo inside-data > /made-inside.txt; cat /made-inside.txt'
echo "host check:"; ls /made-inside.txt 2>&1 || echo "not on host ✔"
echo; echo "=== 4. Memory cap ==="
sudo ./mini.sh --mem 50m /bin/bash -c 'cat /sys/fs/cgroup/memory.max; echo "(want 52428800)"'
echo; echo "=== 5. Net isolation (only lo, like --network none) ==="
sudo ./mini.sh /bin/bash -c 'ip addr 2>/dev/null | head -8 || cat /proc/net/dev | head -8'
