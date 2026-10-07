#!/usr/bin/env bash
set -euo pipefail
NAME="${1:-fsdemo}"
docker rm -f "$NAME" "${NAME}B" fsA fsB fsLayer >/dev/null 2>&1 || true
docker rmi -f lab4-layers:local >/dev/null 2>&1 || true
rm -f Dockerfile.layers
echo "Cleaned up Lab 4."
