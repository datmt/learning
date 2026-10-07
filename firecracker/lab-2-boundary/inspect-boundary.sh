#!/usr/bin/env bash
# Host-side boundary inspection for a running Firecracker VM.
set -u
API_SOCKET="${API_SOCKET:-/tmp/fc-1.socket}"
TAP_DEV="${TAP_DEV:-tap0}"
FC_PID=$(pgrep -o firecracker || true)
[ -z "$FC_PID" ] && { echo "No firecracker process. Boot Lab 1 first."; exit 1; }
echo "== Firecracker PID: $FC_PID =="
ps -o pid,ppid,stat,rss,vsz,cmd -p "$FC_PID"
echo
echo "== /proc/$FC_PID/status (Cap, Seccomp) =="
grep -E '^(Name|State|VmRSS|Cap|Seccomp|Cpus)' "/proc/$FC_PID/status" || true
echo
echo "== Namespaces of VMM =="
lsns -p "$FC_PID" || sudo lsns -p "$FC_PID" || true
echo
echo "== cgroup =="
cat "/proc/$FC_PID/cgroup" || true
echo
echo "== Guest RAM mapping (large anon mapping) =="
grep -E 'rw.*00:00' "/proc/$FC_PID/maps" | awk '{print $1, $6}' | head -5 || true
echo
echo "== Open FDs (socket, tap, rootfs, kernel) =="
ls -l "/proc/$FC_PID/fd" | head -20
echo
echo "== KVM device =="
ls -l /dev/kvm
echo
echo "== TAP device (host side of guest NIC) =="
ip link show "$TAP_DEV" || echo "no $TAP_DEV"
bridge link 2>/dev/null || true
echo
echo "== API version =="
sudo curl -s --unix-socket "$API_SOCKET" http://localhost/version || echo "(API unreachable)"
echo
echo "== Takeaway: ONE host process hosts a whole guest kernel. =="
echo "Guest PIDs, mounts, net stack are invisible here — that IS the boundary."
