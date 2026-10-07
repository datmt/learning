#!/usr/bin/env bash
set -u
echo "=== bridge ==="; bash "$(dirname "$0")/bridge-up.sh"
echo "=== boot A B C ==="
for vm in A B C; do bash "$(dirname "$0")/boot-vm.sh" "$vm" & done
wait
echo "=== mesh test ==="; bash "$(dirname "$0")/mesh-test.sh"
