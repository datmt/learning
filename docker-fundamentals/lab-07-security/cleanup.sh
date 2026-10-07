#!/usr/bin/env bash
set -euo pipefail
echo "Lab 7 needs no teardown (all demo containers were --rm)."
docker ps -a --filter 'name=oomkeep' --format '{{.Names}}' | grep -q . && docker rm -f oomkeep >/dev/null 2>&1 || true
# keep lab7-tools:local (rebuilt automatically if removed):
#   docker rmi -f lab7-tools:local
echo "Clean."
