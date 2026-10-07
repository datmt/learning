#!/usr/bin/env bash
# Same PID-namespace illusion with ZERO Docker — pure Linux.
# Interactive: drops you into a shell that thinks it is PID 1.
set -euo pipefail
echo "==> Your current PID: $$ ; PID ns: $(readlink /proc/self/ns/pid)"
echo "==> Entering: sudo unshare --pid --fork --mount-proc bash"
echo "    Inside, run:  echo \$\$ ; ps aux ; ls /proc   then: exit"
exec sudo unshare --pid --fork --mount-proc bash
