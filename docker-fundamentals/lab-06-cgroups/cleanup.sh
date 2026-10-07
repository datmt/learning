#!/usr/bin/env bash
set -euo pipefail
docker rm -f limited hog oomdemo oomkeep >/dev/null 2>&1 || true
echo "Cleaned up Lab 6."
