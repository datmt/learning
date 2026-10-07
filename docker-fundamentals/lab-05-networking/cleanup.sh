#!/usr/bin/env bash
set -euo pipefail
NAME="${1:-web}"
docker rm -f "$NAME" pingpeer db2 >/dev/null 2>&1 || true
echo "Cleaned up Lab 5."
