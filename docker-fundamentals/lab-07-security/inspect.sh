#!/usr/bin/env bash
# Lab 7 inspect: decode caps to names + security checklist. Read-only.
set -euo pipefail
IMAGE="${IMAGE:-ubuntu:22.04}"
EFF="$(docker run --rm "$IMAGE" awk '/CapEff/{print $2}' /proc/self/status)"
BND="$(docker run --rm "$IMAGE" awk '/CapBnd/{print $2}' /proc/self/status)"
echo "CapEff=$EFF  CapBnd=$BND"
if command -v capsh >/dev/null 2>&1; then capsh --decode="$EFF"; else
  echo "(install libcap2-bin for names)"; python3 -c "
EFF=int('$EFF',16)
names='CHOWN DAC_OVERRIDE DAC_READ_SEARCH FOWNER FSETID KILL SETGID SETUID SETPCAP LINUX_IMMUTABLE NET_BIND_SERVICE NET_BROADCAST NET_ADMIN NET_RAW IPC_LOCK IPC_OWNER SYS_MODULE SYS_RAWIO SYS_CHROOT SYS_PTRACE SYS_PACCT SYS_ADMIN SYS_BOOT SYS_NICE SYS_RESOURCE SYS_TIME SYS_TTY_CONFIG MKNOD LEASE AUDIT_WRITE AUDIT_CONTROL SETFCAP MAC_OVERRIDE MAC_ADMIN SYSLOG WAKE_ALARM BLOCK_SUSPEND AUDIT_READ'.split()
print('kept:',[n for i,n in enumerate(names) if EFF>>i & 1])"; fi
echo; echo "=== checklist for container 'web' (if running) ==="
if docker inspect web >/dev/null 2>&1; then
  docker inspect web --format 'Privileged={{.HostConfig.Privileged}} CapAdd={{.HostConfig.CapAdd}} CapDrop={{.HostConfig.CapDrop}} ReadonlyRootfs={{.HostConfig.ReadonlyRootfs}} Seccomp={{.HostConfig.SecurityOpt}} User={{.Config.User}}'
else
  echo "(no 'web' container — start one from Lab 5 to check, or ignore)"
fi
echo; echo "=== daemon baseline ==="
docker info 2>/dev/null | grep -iE 'security options|seccomp|cgroup version|storage driver|runtimes' || true
