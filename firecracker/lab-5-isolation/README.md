# Lab 5 — Isolation Attacks (Container vs MicroVM)

> Beginner-friendly. The "why microVMs exist" lab. You run the SAME
> benign probes against Docker and Firecracker and watch one family fail.

```text
container → container     shared host kernel, namespaces+cgroups filter only
microVM → microVM         separate guest kernels, KVM hardware boundary
```

## 0. How isolation works (the 4 Linux mechanisms + the 5th)

Containers combine 4 kernel features. Each is a *filter in the shared
kernel* — fast, but a kernel bug bypasses all of them at once:

| Mechanism | What it does | Everyday analogy | Check it |
|---|---|---|---|
| **Namespaces** (`lsns`, `unshare`) | Give a process its own view: PIDs, mounts, network, hostnames. | Frosted glass — you see only your office. | `lsns -p <pid>` |
| **cgroups** (`/sys/fs/cgroup`) | Limit/share CPU, RAM, PIDs. "You get 100 MB, 0.5 CPU." | Office ration card. | `cat /proc/<pid>/cgroup` |
| **Capabilities** (`CAP_*`) | Split root's powers: `CAP_NET_ADMIN` (network), `CAP_SYS_ADMIN` (mount!), `CAP_SYS_PTRACE`… Docker drops most by default; `--privileged` restores all (danger). | Master key cut into pieces. | `grep Cap /proc/<pid>/status` + `capsh --decode` |
| **Seccomp** | Syscall allowlist: "this container may call these 200 syscalls, never `mount`/`reboot`". | Bouncer with a guest list. | `grep Seccomp /proc/<pid>/status` |
| **KVM boundary (microVMs only)** | Guest syscalls never reach the host kernel — handled by the *guest* kernel running on virtual hardware. The host only sees one VMM process doing ioctls. | Separate building, not just frosted glass. | Lab 2's `maps` + `STOP`-freeze demo |

Mental rule: **containers share the kernel; microVMs share only the
hardware.** Everything in the attack matrix flows from that sentence.

## 1. The attacks, translated for beginners

`attack-matrix.sh` runs only **benign probes** (no exploits, no CVEs).
Each one answers "can tenant X see/affect tenant Y or the host?":

1. **`uname -r` — do we share a kernel?** Container: prints the *host*
   kernel version (proof: same kernel). MicroVM: prints the *guest*
   kernel (different version — proof of separation).
2. **`/proc` snooping — can I see others' processes?** Container:
   `ps aux` shows a filtered but host-shaped list; misconfigurations leak.
   MicroVM: only guest PIDs exist; host PIDs are unrepresentable.
3. **`dmesg` / `insmod` — can I touch the kernel?** Container shares the
   kernel log and module loader — blocked only by caps/seccomp (fragile).
   MicroVM has its *own* log and module space; loading a guest module
   affects nobody else.
4. **Net sniff — can I see others' traffic?** On a shared Docker bridge,
   a `CAP_NET_RAW` container can tcpdump neighbors. MicroVMs each have a
   private TAP; frames never cross without the bridge forwarding them,
   and the guest has no promiscuous path to another TAP.
5. **Fork bomb (`ulimit`) — what's the blast radius?** Container: eats
   host scheduler/cgroup until limits kick in — noisy neighbor. MicroVM:
   eats *guest* scheduler; host sees one busy VMM process. `kill -STOP`
   the VMM and the whole storm pauses.
6. **`mount` escape — can I grab host files?** Classic container escape:
   with `CAP_SYS_ADMIN` (or a leaked `/var/run/docker.sock`), mount the
   host disk inside the container. MicroVM: there is no host mount to
   grab — the only disk is the virtio `.ext4` file.

## 2. Files

- `commands.sh` — runbook (docker victims → boot 2 VMs → matrix → write-up).
- `docker-victim.sh` — starts `tenant-a`/`tenant-b` (plain `alpine sleep`,
  no `--privileged` — fair default comparison). Skips cleanly if Docker
  is absent.
- `attack-matrix.sh` — read-only probes in both worlds, `head`-truncated
  output. Safe to run repeatedly.
- `results-template.md` — the actual deliverable: 6-row table + one
  paragraph ("when would I pick containers vs microVMs?").

## 3. Quick start

```bash
cd lab-5-isolation
bash docker-victim.sh        # needs docker; skip if unavailable
bash attack-matrix.sh        # needs 2 running microVMs (see Lab 4)
```

Write up: for each attack, `container: ESCAPED / CONTAINED`,
`microVM: ESCAPED / CONTAINED`, one-line why. There are no trick answers
— "container contained it *because of seccomp*" is a correct, useful row.

## 4. Expected outcome (spoiler — confirm it yourself)

| Attack | Container | MicroVM | Why |
|---|---|---|---|
| kernel version (`uname`) | shared kernel | own kernel | KVM boundary |
| /proc walk | filtered view, host-shaped | invisible | separate kernel + PID space |
| `CAP_SYS_ADMIN` mount escape | possible if misconfigured | N/A — no host mounts exist | no shared mounts |
| net sniff | possible on shared bridge | only own TAP | L2 isolation per TAP |
| fork bomb | cgroup-limited, still host scheduler | guest scheduler only | nested scheduling |

Honest footnote: containers *usually* contain these too when configured
well (dropped caps + seccomp + user ns). The lesson is **defense in
depth and blast radius**: one kernel bug breaks the container filter for
everyone; the same bug in a guest breaks one guest.

## Safety

All probes are **non-destructive, local-only**. Do NOT download/run
kernel exploits you don't understand. If a probe ever looks dangerous,
read it first — every command in `attack-matrix.sh` is printed before it
runs and truncated to 8 lines.

## Verify

- `results-template.md` filled: 6 rows with ESCAPED/CONTAINED + one-line why.
- One paragraph at the bottom: "I would pick containers when ___, microVMs
  when ___." (Hint: startup ms + density vs tenant trust + kernel freedom.)
- Cleanup: `sudo docker rm -f tenant-a tenant-b`; shut VMs down via Lab 4 notes.
