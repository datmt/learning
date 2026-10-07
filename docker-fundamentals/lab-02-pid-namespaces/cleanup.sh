#!/usr/bin/env bash
set -euo pipefail
NAME="${1:-piddemo}"
docker rm -f "$NAME" >/dev/null 2>&1 || true
echo "Cleaned up '$NAME'."
