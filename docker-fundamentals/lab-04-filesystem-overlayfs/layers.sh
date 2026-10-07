#!/usr/bin/env bash
# Build a tiny 3-layer image so you can WATCH layers appear in docker history.
# Usage: ./layers.sh  (leaves image lab4-layers:local)
set -euo pipefail
cd "$(dirname "$0")"
cat > Dockerfile.layers <<'EOF'
FROM ubuntu:22.04
RUN echo one > /one.txt
RUN echo two > /two.txt && mkdir -p /app && echo v1 > /app/ver.txt
EOF
docker build -f Dockerfile.layers -t lab4-layers:local . 
echo; echo "=== history of lab4-layers:local ==="
docker history lab4-layers:local --format 'table {{.CreatedBy}}\t{{.Size}}'
echo; echo "Tip: docker run --rm lab4-layers:local ls -l /one.txt /two.txt /app/ver.txt"
