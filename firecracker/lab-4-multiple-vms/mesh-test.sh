#!/usr/bin/env bash
# Full mesh ping test across A/B/C.
set -u
KEY="$(ls "$HOME"/fc-lab1/ubuntu-*.id_rsa 2>/dev/null | grep -v pub | tail -1)"
declare -A IP=( [A]=10.0.0.2 [B]=10.0.0.3 [C]=10.0.0.4 )
for src in A B C; do
  for dst in A B C 10.0.0.1 8.8.8.8; do
    [ "$src" = "$dst" ] && continue
    target="${IP[$dst]:-$dst}"
    if ssh -i "$KEY" -o StrictHostKeyChecking=no -o ConnectTimeout=5 "root@${IP[$src]}" "ping -c2 -W2 $target" >/dev/null 2>&1; then
      echo "PASS: $src(${IP[$src]}) -> $target"
    else
      echo "FAIL: $src(${IP[$src]}) -> $target"
    fi
  done
done
echo "--- bridge ---"; bridge link || ip link show type bridge
