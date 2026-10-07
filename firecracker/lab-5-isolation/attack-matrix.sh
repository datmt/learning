#!/usr/bin/env bash
# Benign isolation probes: Docker vs microVM. No exploits, read-only + limits.
set -u
KEY="$(ls "$HOME"/fc-lab1/ubuntu-*.id_rsa 2>/dev/null | grep -v pub | tail -1)"
VM_A="${VM_A:-10.0.0.2}"; VM_B="${VM_B:-10.0.0.3}"
have_vm=false; [ -n "${KEY:-}" ] && have_vm=true
have_docker=false; command -v docker >/dev/null && sudo docker ps >/dev/null 2>&1 && have_docker=true

probe() { echo "--- $1 ---"; shift; eval "$@" 2>&1 | head -8; echo; }

echo "### HOST baseline ###"
probe "host kernel" "uname -r"
$have_docker && {
  echo "### DOCKER tenant-a ###"
  probe "container kernel (== host? BAD for isolation)" "sudo docker exec tenant-a uname -r"
  probe "container sees host procs?" "sudo docker exec tenant-a ps aux | head -8"
  probe "container caps" "sudo docker exec tenant-a cat /proc/self/status | grep -i cap"
}
$have_vm && {
  echo "### MICROVM-A ###"
  SSH="ssh -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=5 root@$VM_A"
  probe "guest kernel (!= host? GOOD)" "$SSH 'uname -r'"
  probe "guest PID1" "$SSH 'ps -p 1 -o comm,args'"
  probe "guest seeks host VMM (should find nothing)" "$SSH 'ps aux | grep -c firecracker || echo 0'"
  probe "guest /proc/cmdline" "$SSH 'cat /proc/cmdline'"
  probe "guest tries to see VM-B disk (should fail)" "$SSH 'ls /dev/vd* /dev/sd* 2>&1'"
} || echo "(no microVMs running — boot Lab 4 first)"
echo "Record verdicts in results-template.md."
