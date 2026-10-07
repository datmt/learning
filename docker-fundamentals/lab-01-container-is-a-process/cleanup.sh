#!/usr/bin/env bash
# Idempotent teardown for Lab 1.
set -euo pipefail
NAME="${1:-demo}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "Cleaned up container '$NAME' (if it existed)."
