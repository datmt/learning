#!/usr/bin/env bash
# Client examples + tiny benchmark for Lab 7.
set -u
PORT="${PORT:-8080}"
base="http://localhost:$PORT"
echo "== health =="; curl -s "$base/health"; echo
echo "== single run =="; curl -s -X POST "$base/run" -d '{"cmd":"uname -a; echo hello-from-microvm"}'; echo
echo "== 3 sequential (cold start each) =="
for i in 1 2 3; do
  /usr/bin/time -f "wall %es" curl -s -X POST "$base/run" -d '{"cmd":"echo job '$i'"}' 2>&1; echo
done
echo "== stats =="; curl -s "$base/stats"; echo
