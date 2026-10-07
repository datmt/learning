# Lab 2 — Understand the VM Boundary

> Beginner-friendly. Assumes Lab 1 booted. Goal: *see* the isolation
> boundary with your own tools, then break things safely to prove where
> it is.

```text
Host
 ├── firecracker (one normal userspace process, has a PID)
 ├── /dev/kvm (door to the host kernel's KVM module, via ioctls)
 ├── tap0 (host end of the virtual Ethernet cable)
 └── VM (guest RAM = anonymous mmap inside the firecracker process)

Guest (invisible from host ps/lsns — that invisibility IS the boundary)
 ├── kernel (own page tables, handles its own syscalls)
 ├── init / processes (PIDs start at 1 again)
 └── network (own stack: eth0 via virtio-net → tap0)
```

## 0. What "boundary" even means

When people say "container boundary" vs "VM boundary", they mean: **what
must an attacker break to reach the host or a neighbor?**

```text
Container:  app → (namespaces + cgroups filter) → SAME host kernel
MicroVM:    app → guest kernel → KVM hardware boundary → host kernel
```

- A **namespace** is a kernel filter that says "you only see *your*
  PIDs / mounts / network". It is still one kernel enforcing the filter.
  A kernel bug can bypass it.
- A **microVM** runs a *second* kernel under hardware virtualization
  (AMD-V on your machine). Host syscalls from the guest don't exist —
  the guest's `read()` is handled by the *guest* kernel. To touch the
  host, you must escape the virtual hardware *and* the Firecracker
  process. That is a much smaller attack surface (Firecracker ≈ 50k
  lines vs full QEMU + kernel sharing).

This lab makes that abstract claim concrete with 5 questions.

## 1. Vocabulary

| Word | Meaning | Why it matters here |
|---|---|---|
| **PID** | Process ID. Host and guest each start counting at 1. Guest PID 1 (`init`) is *not* host PID 1 — it lives inside the VM. | Proves separate process tables. |
| **`/proc/<pid>`** | Fake filesystem the kernel generates per process: `maps` (memory), `status` (caps/seccomp), `fd/` (open files). Your main microscope. | `maps` shows guest RAM as one big host mapping. |
| **`lsns`** | Lists namespaces a process belongs to. Containers = same kernel, different ns. Firecracker = barely any ns — isolation comes from KVM, not ns. | Surprise moment: VMM has *few* namespaces, yet isolation is stronger. |
| **Capabilities (`Cap*`)** | Fine-grained root powers (`CAP_NET_ADMIN`, `CAP_SYS_ADMIN`). `/proc/<pid>/status` shows which the VMM holds. | Fewer caps = smaller blast radius. |
| **Seccomp** | Syscall firewall: which syscalls a process may call. Firecracker locks itself down after boot. | Even if VMM is tricked, it can't call arbitrary host syscalls. |
| **cgroup** | Resource bucket (CPU shares, memory max, PIDs max). Containers rely on it heavily; Firecracker uses fixed `mem_size_mib` + host scheduler. | Fork bomb behaves differently in each (Lab 5). |
| **FD (file descriptor)** | Open handle: socket, TAP, rootfs file, log. `ls /proc/<pid>/fd` shows everything the VMM holds open. Kill the VMM → all close → VM vanishes. | Explains why `kill -9` = instant power-off. |
| **Signal (`STOP/CONT/KILL`)** | `STOP` freezes a process, `CONT` resumes, `KILL` destroys. Freezing the VMM freezes the *entire guest clock*. | Coolest demo: guest `date` loop pauses while host is fine. |
| **ioctl → `/dev/kvm`** | How Firecracker asks the host kernel to run guest code on real CPU with memory translation (EPT/NPT page tables). | The only "magic" — everything else is normal files/sockets. |
| **virtio-net** | Paravirtual NIC: guest thinks it has Ethernet hardware; Firecracker shuttles frames to `tap0`. No emulated Realtek card. | Why deleting `tap0` kills net but not the VM. |

## 2. The 5 questions, explained

1. **Host shows ONE `firecracker` process; guest `ps aux` shows dozens.**
   Because the guest kernel schedules *its own* processes internally.
   The host kernel only ever scheduled one thing: the VMM. The guest's
   `init`, `sshd`, `bash` are invisible to host `ps` — they are just
   guest-RAM bytes being interpreted by KVM.
2. **`lsns -p <vmm-pid>` vs `lsns` in guest.** Host side: VMM sits in
   host namespaces (maybe one net/mnt ns if jailer used). Guest side:
   a whole fresh namespace set owned by the guest kernel. No overlap.
3. **Guest can't read host `/proc` or see `tap0`.** Guest `open()` goes
   to the *guest* kernel, whose `/proc` only knows guest PIDs. `tap0`
   lives on the host; the guest only sees its own `eth0`. There is no
   path between them except Ethernet frames.
4. **`kill -9 <vmm-pid>` kills the VM, host unaffected.** You destroyed
   the landlord process; guest RAM (its `mmap`) is freed. Host kernel
   never depended on the guest — it just scheduled it.
5. **Deleting `tap0` kills net, not the VM.** The NIC backend vanished,
   but vCPU/RAM/disk are independent devices. Proves Firecracker's
   device model: tiny set of independent virtio devices, no giant
   emulated motherboard.

## 3. Files

- `commands.sh` — guided runbook (inspect → probe → break).
- `inspect-boundary.sh` — host-side microscope: `ps`, `/proc/<pid>/status`
  (Caps/Seccomp), `lsns`, `cgroup`, `maps` (find the ~256 MB anon mapping
  = guest RAM), `fd/` (socket, TAP, rootfs, log), TAP + API version.
- `guest-probe.sh` — SSH into guest, compare `uname -a`, PID 1, `ps`,
  mounts, `ip route`, `/proc/cpuinfo`, prove host VMM invisible from inside.
- `break-it.sh` — interactive chaos menu (STOP-freeze, del-TAP, fill disk,
  graceful `reboot`). All reversible except `kill -9` (by design).

## 4. Quick start

```bash
cd lab-2-boundary
bash ../lab-1-boot/21-boot-ssh.sh &   # boot one VM first (or reuse running one)
bash inspect-boundary.sh
bash guest-probe.sh
bash break-it.sh   # pick 1-4, watch host vs guest impact
```

Try this freeze demo manually — it always lands:

```bash
# terminal A (guest): watch time tick
ssh -i ~/fc-lab1/ubuntu-*.id_rsa root@172.16.0.2 'while true; do date; sleep 1; done'
# terminal B (host): freeze the whole VM for 5s
sudo kill -STOP $(pgrep -o firecracker); sleep 5; sudo kill -CONT $(pgrep -o firecracker)
# terminal A resumes — note the 5s GAP. Host clock never stopped.
```

## 5. Verify (what each check proves)

- Host `cat /proc/<fc-pid>/maps` shows a large anonymous mapping ≈ guest
  RAM (e.g. 256 MB). Proves guest memory is just host-process memory.
- Guest `dmesg | head` shows `Booting Linux` under KVM, `systemd`/init as
  PID 1. Proves a real kernel booted, not a chroot trick.
- `kill -STOP` freezes guest clock, host unaffected. Proves host
  schedules the VM as one entity — guest scheduler is nested inside.

## Cleanup

```bash
ssh -i ~/fc-lab1/ubuntu-*.id_rsa root@172.16.0.2 reboot  # graceful
sudo ip link del tap0 2>/dev/null || true
sudo rm -f /tmp/fc-1.socket
```

`kill -9 $(pgrep firecracker)` is the crash cord — fine in lab, but
prefer `reboot` so rootfs unmounts cleanly.
