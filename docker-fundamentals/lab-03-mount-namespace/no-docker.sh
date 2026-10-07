#!/usr/bin/env bash
# Pure-Linux mount isolation, no Docker. Interactive.
set -euo pipefail
mkdir -p /tmp/mntdemo-lab
echo "hello from $(hostname)" > /tmp/mntdemo-lab/file.txt
echo "Created /tmp/mntdemo-lab/file.txt. Entering: sudo unshare --mount --fork bash"
echo "Inside, run:"
echo "  mount --bind /tmp/mntdemo-lab /mnt && ls /mnt     # visible HERE only"
echo "  # open a 2nd terminal: ls /mnt  -> empty there.  Then: exit"
exec sudo unshare --mount --fork bash
