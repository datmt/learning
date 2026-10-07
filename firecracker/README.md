# Firecracker Hands-On Labs — Network/Security Architecture Track

Goal: learn **network + security architecture**, not just app code.
Each lab is 1 folder with `README.md` (theory + steps), `commands.sh`
(copy-paste runnable), and supporting code/configs.

Detected host for these labs:

```text
Arch Linux, x86_64, bare metal, AMD-V, /dev/kvm present (crw-rw-rw-)
```

> Your original brief assumed Ubuntu/ARM64 (GB10/DGX). These labs
> auto-detect `ARCH=$(uname -m)` so they work on **both x86_64 and aarch64**.
> On ARM64 add `keep_bootcon` to kernel boot args (scripts already do this).

## Lab map

```text
lab-1-boot        Boot a microVM (kernel + rootfs + serial console)
lab-2-boundary    Understand the VM boundary (host vs guest, break things)
lab-3-networking  TAP + NAT + routing: microVM -> host -> Internet
lab-4-multiple-vms  Bridge 3 microVMs: VM-A <-> VM-B <-> VM-C
lab-5-isolation   container vs microVM escape/isolation attacks
lab-6-launcher    Tiny VM launcher (API socket, Python)
lab-7-serverless  Mini serverless: POST /run -> VM per job
```

## Prerequisites (all labs)

```bash
# Arch:
sudo pacman -S --needed curl wget jq iproute2 iptables nftables \
  e2fsprogs squashfs-tools openssh iperf3 python3 docker --noconfirm

# Ubuntu (if on DGX/GB10):
# sudo apt install -y curl wget jq iproute2 iptables nftables \
#   e2fsprogs squashfs-tools openssh-client iperf3 python3 docker.io

[ -r /dev/kvm ] && [ -w /dev/kvm ] && echo "KVM OK" || echo "KVM FAIL"
lsmod | grep kvm
```

Firecracker binary install is covered in `lab-1-boot/commands.sh`.
Pinned reference version: `v1.16.1` (scripts default to `latest` if unset).

## Suggested order

1. `lab-1-boot` — get the "Welcome to Linux" moment.
2. `lab-2-boundary` — `ps`, `/proc`, namespaces, seccomp, kill -9 the VMM.
3. `lab-3-networking` — single VM with Internet. This is the core net lab.
4. `lab-4-multiple-vms` — virtual network + inter-VM ping/iperf.
5. `lab-5-isolation` — why microVMs exist (compare with Docker).
6. `lab-6-launcher` — automate labs 1–4 via code.
7. `lab-7-serverless` — capstone: VM-per-request platform.

Each lab README ends with Verify + Cleanup + Troubleshooting. Always run
cleanup before the next lab (`ip link del`, `rm -f /tmp/fc-*.socket`).
