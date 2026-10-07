#!/usr/bin/env bash
# Lab 3 runbook.
set -u
echo "=== 1. Host network up ==="
bash "$(dirname "$0")/host-net.sh" up
echo "=== 2. Boot VM with network (if not running) ==="
echo "bash ../lab-1-boot/21-boot-ssh.sh"
echo "=== 3. Guest network (if serial-booted, run inside guest) ==="
echo "bash guest-net.sh  # or: ssh root@172.16.0.2 < guest-net.sh"
echo "=== 4. Verify ==="
bash "$(dirname "$0")/verify-net.sh" || echo "(boot VM first, then re-run verify)"
