# Lab 1 — Boot a MicroVM

> No low-level background assumed. This README teaches the fundamentals
> first, then you boot. Read it once top-to-bottom; the commands will
> make sense afterwards.

## 0. The big picture in 30 seconds

Your laptop runs **one Linux kernel** (the host kernel). Normally every
program you start (`firefox`, `python3`, `docker run ...`) asks *that*
kernel for things via **syscalls**: "give me memory", "write this file",
"send this packet".

A Firecracker microVM adds a second kernel:

```text
Docker container (for comparison)
────────────────────────────
Host kernel
 ├── process (your app, isolated by namespaces + cgroups)
 └── ...same kernel handles its syscalls...

Firecracker microVM (this lab)
────────────────────────────
Host Linux
 └── Firecracker process (a normal userspace program YOU start)
      └── KVM (/dev/kvm — asks host kernel to fake hardware)
           └── Guest Linux kernel (its OWN kernel, own syscalls)
                └── init → login → your app
```

So the syscall path changes:

```text
container:  app → host kernel → hardware
microVM:    app → guest kernel → virtual hardware (KVM) → host kernel → hardware
```

That extra layer is the entire point: even if the app compromises its
*guest* kernel, it is still trapped inside a host userspace process.
Labs 2 and 5 prove this by attacking both.

---

## 1. Vocabulary (read this if words feel blurry)

| Word | Plain-English meaning | Analogy |
|---|---|---|
| **Kernel** | The core program that controls CPU, RAM, disk, network. Everything else asks it for help. | Building manager — tenants (apps) file requests, manager holds keys. |
| **Syscall** | A request from an app to the kernel (`read`, `write`, `fork`). | Filling out a form at reception. |
| **Process** | One running program (has PID, memory, file descriptors). Firecracker itself is just a process. | One tenant in the building. |
| **vCPU / RAM** | Slices of host CPU time + a chunk of host memory *pretending* to be a physical computer. Guest RAM is just `mmap`'d memory inside the Firecracker process (Lab 2 shows this). | Reserving 1 meeting room + 256 MB of whiteboard. |
| **vmlinux** | An *uncompressed* Linux kernel binary (ELF format). Firecracker loads this directly into guest RAM and jumps to it. Newer Firecracker also accepts `bzImage` (compressed x86 kernel). | The OS installer CD. |
| **rootfs / `.ext4`** | A file on your host that *contains a whole filesystem* (binaries, `/etc`, `/root`). Firecracker exposes it as a virtual hard disk. `ext4` is just Linux's standard filesystem format. | A hard-drive image in a single file. |
| **Block device / drive** | Something that looks like a disk (`/dev/vda`). In the guest, your `.ext4` file appears as one. | Plugging a USB stick into the VM. |
| **boot args** | Text string passed to the guest kernel: `console=ttyS0 reboot=k panic=1`. Tells it where to print boot logs, what to do on reboot/panic. | Kernel command-line cheat sheet. |
| **`init` / PID 1** | First program the guest kernel starts. It then starts everything else (login prompt, sshd, systemd). Host and guest each have their own PID 1. | First employee who unlocks the branch office. |
| **Serial console (`ttyS0`)** | The simplest possible text display: guest kernel prints characters, Firecracker copies them to your terminal. No graphics, no network needed. Earliest thing that works. | A walkie-talkie into the VM. |
| **TAP device (`tap0`)** | A virtual Ethernet cable. Host end is `tap0`, guest end is `eth0`. Ethernet frames written on one end appear on the other. | A virtual patch cable between host and guest. |
| **MAC / IP** | MAC = hardware address on the cable (e.g. `06:00:AC:10:00:02`). IP = logical address for routing (e.g. `172.16.0.2`). You need both for SSH to work. | MAC = serial number on NIC; IP = postal address. |
| **SSH + keypair** | Encrypted remote login. `id_rsa` (private, stays on host) + `id_rsa.pub` (public, baked into guest `authorized_keys`). Guest `sshd` only lets in whoever holds the private key. | Key + lock. We manufacture both in `10-fetch-assets.sh`. |
| **KVM / `/dev/kvm`** | Kernel feature that lets a userspace program create VMs (AMD-V/Intel-VT hardware assist). If you can't read/write `/dev/kvm`, you can't run Firecracker. | The building permit for fake computers. |
| **VMM** | Virtual Machine Monitor — the userspace program managing the VM. Here: the `firecracker` binary. QEMU is another VMM (huge); Firecracker is tiny (~50k lines). | The landlord of fake computers. |
| **API socket** | A Unix socket file (`/tmp/fc-1.socket`) that Firecracker listens on. You configure the VM by sending it HTTP `PUT` requests with `curl`. Not TCP — a local file. | A service hatch on the VMM — you slide JSON orders through it. |
| **`jailer`** | Firecracker's sandbox wrapper (chroot + cgroups + dropped privileges). We skip it in Lab 1 for clarity; Lab 7 stretch goal enables it. | A prison cell *around* the landlord. |

---

## 2. What Firecracker needs (and why each piece exists)

Minimum to boot:

```text
firecracker
   │
   ├── vCPU: 1          → how many virtual CPUs the guest sees
   ├── RAM: 256 MB      → how big the guest's physical memory is
   ├── kernel: vmlinux  → WHAT to boot (the guest OS code itself)
   └── rootfs: rootfs.ext4 → what disk to mount as / (userspace: /bin, /etc, sshd)
```

- **Without vCPU/RAM**: the fake hardware has no CPU to execute or RAM to
  execute *in*. `machine-config` sets this.
- **Without kernel**: there is nothing to start. Firecracker is *not* an
  emulator that boots BIOS → bootloader → OS. It loads the kernel image
  straight into RAM (fast: ~100 ms).
- **Without rootfs**: the kernel would boot, panic, and die (`VFS: Unable to
  mount root fs`). The kernel alone can't give you a login prompt — it needs
  `/sbin/init`, `/bin/sh`, `/etc/passwd` from the rootfs.

Optional but needed for networking (Lab 3):

```text
   ├── TAP: tap0        → virtual cable
   ├── MAC: 06:00:...   → guest NIC identity
   └── Guest IP: 172.16.0.2 → so host and guest can route packets
```

---

## 3. How a boot actually happens (follow along with the scripts)

```
you run ./firecracker --api-sock /tmp/fc-1.socket
   → VMM starts, listens on that socket file, does NOTHING else yet
   → (this is why curl to a dead socket = "Couldn't connect")

PUT /logger            → "write debug logs to ./fc-1.log"
PUT /boot-source       → "kernel lives HERE, boot with THESE args"
PUT /machine-config    → "1 vCPU, 256 MB RAM"
PUT /drives/rootfs     → "attach THIS .ext4 file as /"
PUT /network-interfaces (SSH variant only) → "plug virtio-net NIC to tap0"
PUT /actions {"InstanceStart"} → "GO: power on the virtual CPU"

guest kernel decompresses/starts → prints to ttyS0 → you see "Booting Linux"
   → mounts rootfs → starts init → login prompt (serial) or sshd (SSH variant)
```

Two ways to trigger the same sequence:

| Method | How | When to use |
|---|---|---|
| **API socket** (`20-boot-serial.sh`, `21-boot-ssh.sh`) | 5–6 `curl --unix-socket` calls, then `InstanceStart` | Learning (you see every knob). Default in this lab. |
| **Config file** (`configs/vm-config.json`) | One JSON + `--config-file`, VM auto-starts | Automation (Lab 6 compares both). |

---

## 4. Serial console vs SSH — which file does what

### `20-boot-serial.sh` — start here (no network, fewest moving parts)

- Configures kernel + RAM + disk only. **No TAP, no IP, no SSH.**
- Guest prints boot log to `console=ttyS0`, which Firecracker mirrors to
  your terminal. You log in as `root` / `root` right there.
- Type `reboot` in the guest → Firecracker exits (it has no power button,
  so guest reboot = VMM shutdown).
- Use this to prove "I booted a real second kernel" with minimum fuss.

### `21-boot-ssh.sh` — second step (adds virtual networking, bridge to Lab 3)

This is the same boot **plus** L2/L3 plumbing so you can `ssh` in:

1. **Host makes the cable**: `ip tuntap add dev tap0 mode tap`,
   assigns host end `172.16.0.1/30`, brings it `up`.
2. **Host becomes a router**: `echo 1 > /proc/sys/net/ipv4/ip_forward`
   plus one `iptables -t nat ... MASQUERADE` rule so guest packets can
   leave to the Internet with the host's source IP.
3. **VMM plugs the NIC**: `PUT /network-interfaces/net1` with
   `guest_mac=06:00:AC:10:00:02` + `host_dev_name=tap0`.
   Convention: last byte `02` ↔ guest IP `.2`. If MAC and IP disagree,
   ARP breaks and SSH hangs — keep them in sync.
4. **Guest configures its end** (over first SSH): give `eth0`
   `172.16.0.2/30`, `ip route add default via 172.16.0.1`,
   write `/etc/resolv.conf` (8.8.8.8) for DNS.
5. **You log in**: `ssh -i ubuntu-*.id_rsa root@172.16.0.2`.
   The keypair was minted in `10-fetch-assets.sh`, public half baked into
   the guest image, private half stays on your host.

If serial is a walkie-talkie, SSH is a phone call: more setup (cable +
addresses + keys), but you get a real network path you can `ping`,
`curl`, and `iptables`-filter in later labs.

`/30` means "4 addresses, 2 usable" (`.0` network, `.1` host, `.2` guest,
`.3` broadcast). Deliberately tiny — Lab 4 upgrades to a `/24` bridge
when one cable is no longer enough.

---

## 5. Files

- `commands.sh` — full copy-paste runbook (install → boot → shutdown).
  Read it first; it just calls the numbered scripts below in order.
- `00-install.sh` — downloads the `firecracker` binary for your
  `ARCH` (`x86_64` here, `aarch64` on GB10/DGX) at pinned `v1.16.1`
  (override with `FC_VERSION=latest`). Verifies `--version`.
- `10-fetch-assets.sh` — fetches the guest kernel (`vmlinux-*`) and an
  Ubuntu rootfs (`.squashfs` → unpack → inject your fresh SSH key →
  repack as `.ext4`). Ends with `e2fsck` sanity check.
- `20-boot-serial.sh` — boot with serial console only (no network).
  Best first boot. Blocking: your terminal becomes the guest console.
- `21-boot-ssh.sh` — boot with TAP + SSH (mini Lab 3 built in).
  Creates `tap0`, NAT, boots, configures guest net, drops you in SSH.
- `configs/vm-config.json` — same VM as JSON for
  `--config-file` mode: `sudo ./firecracker --api-sock ... --config-file
  configs/vm-config.json`. Edit the two absolute paths first.

## 6. Quick start

```bash
cd lab-1-boot
cat commands.sh   # read first
bash 00-install.sh
bash 10-fetch-assets.sh
bash 20-boot-serial.sh   # terminal becomes guest console
# login as root/root, then type `reboot` to exit
```

Then try the SSH variant:

```bash
bash 21-boot-ssh.sh
# ssh -i ~/fc-lab1/ubuntu-*.id_rsa root@172.16.0.2
```

Config-file variant (compare with API mode above):

```bash
# edit configs/vm-config.json paths to your $HOME/fc-lab1 files, then:
sudo ./firecracker --api-sock /tmp/fc-1.socket \
  --config-file configs/vm-config.json
```

## 7. Verify (and what each check proves)

- `20-boot-serial.sh` prints guest kernel boot logs to your terminal.
  Proves the guest kernel executed (a container never prints this).
- `curl --unix-socket /tmp/fc-1.socket http://localhost/version`
  returns JSON. Proves the VMM control plane is alive.
- Host `uname -a` vs guest `uname -a` show **different kernel versions**.
  Proves separate kernels (in Docker they'd be identical).
- `pgrep -a firecracker` shows exactly **one** host process per VM.
  Proves the whole guest lives inside one host process (Lab 2 dissects it).

## Cleanup

```bash
sudo rm -f /tmp/fc-1.socket /tmp/fc-1.log
sudo ip link del tap0 2>/dev/null || true
```

Graceful shutdown is always `reboot` *inside* the guest, or
`SendCtrlAltDel` via API. `kill -9` the Firecracker PID works but is the
crash-power-off (Lab 2 uses it deliberately).

## Troubleshooting

- `KVM FAIL` (`[ -r /dev/kvm ]` fails) → lab shortcut
  `sudo chmod 666 /dev/kvm`, proper fix: add yourself to the `kvm` group
  or use `setfacl -m u:${USER}:rw /dev/kvm`. Without this, KVM ioctls are
  denied and Firecracker exits immediately.
- `curl: (7) Couldn't connect` → Firecracker process died; check
  `./fc-1.log` (usually bad kernel/rootfs path or socket already in use —
  `sudo rm -f /tmp/fc-1.socket`).
- No boot output / instant exit → wrong-`ARCH` kernel (x86_64 kernel on
  aarch64 host or vice versa). Re-run `10-fetch-assets.sh` on the same
  machine you boot on; it uses `uname -m` to pick correctly.
- SSH hangs at `connecting` → MAC↔IP mismatch or guest route missing.
  Re-check `FC_MAC=...:02` ↔ `172.16.0.2`, host TAP `up`, forwarding `=1`.
- Slow DNS inside guest but IPs ping fine → missing
  `options single-request-reopen` in guest `resolv.conf` (IPv6 AAAA stall
  behind NAT). `21-boot-ssh.sh` / `guest-net.sh` already append it.
- On aarch64 add `keep_bootcon` (scripts do this automatically via
  `uname -m`). Without it early boot logs vanish on ARM.
