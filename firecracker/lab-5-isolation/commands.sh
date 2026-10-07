#!/usr/bin/env bash
set -u
echo "=== 1. Docker comparison (optional) ==="
bash "$(dirname "$0")/docker-victim.sh" || echo "(docker unavailable — microVM-only mode)"
echo "=== 2. Boot 2 microVMs (Lab 4 subset) ==="
echo "cd ../lab-4-multiple-vms && bash bridge-up.sh && bash boot-vm.sh A & bash boot-vm.sh B & wait"
echo "=== 3. Attack matrix ==="
bash "$(dirname "$0")/attack-matrix.sh"
echo "=== 4. Record results in results-template.md ==="
