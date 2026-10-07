#!/usr/bin/env bash
# Pure-Linux net isolation. Interactive.
set -euo pipefail
echo "Entering: sudo unshare --net --fork bash"
echo "Inside run:  ip addr     # only lo — no eth0, no route out"
echo "             ping -c1 8.8.8.8   # fails: no veth/bridge/NAT. Then: exit"
exec sudo unshare --net --fork bash
