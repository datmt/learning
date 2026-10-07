# Lab 4 — Multiple MicroVMs on a Virtual Network

> Beginner-friendly. Lab 3 was one cable; now you build a virtual
> **switch** and plug 3 VMs into it. This is the smallest realistic
> cloud network.

```text
                    Host
                     │
          ┌──────────┼──────────┐
          │          │          │
        tap-a      tap-b      tap-c      (one TAP per VM — never share)
       VM-A       VM-B       VM-C
     10.0.0.2   10.0.0.3   10.0.0.4
     MAC :02    MAC :03     MAC :04
          │          │          │
          └──────────┼──────────┘
                     │ fc-br0 (bridge = software switch, 10.0.0.1/24)
                  virtual
                  network
```

## 0. Why a bridge? (Lab 3 doesn't scale)

Lab 3's `/30` fits exactly 1 guest. For N guests you need:

1. **A switch, not a cable.** A Linux **bridge** (`fc-br0`) is a software
   Ethernet switch: it learns which MAC lives on which TAP port and
   forwards frames port-to-port. `bridge link` shows enslaved ports;
   `bridge fdb` shows the learned MAC table.
2. **A bigger subnet.** `/24` = 256 addresses (`.0` net … `.255`
   broadcast, 254 usable). Room for A–C plus headroom for Lab 7 jobs.
3. **One gateway IP on the bridge** (`10.0.0.1`). Guests use it as
   default route; host NATs it to the Internet (same MASQUERADE as Lab 3).
4. **Strict 1:1:1:1 mapping per VM** — socket, TAP, MAC, IP, rootfs copy
   must ALL be unique (table below). Sharing any one causes spooky bugs.

| Must be unique | Why sharing breaks |
|---|---|
| API socket (`/tmp/fc-a.socket`) | Two VMMs can't listen on one file; second boot fails. |
| TAP (`tap-a`) | Two guests on one cable = MAC fight, packets to wrong VM. |
| MAC | Switch learns MAC→port; duplicates flap the table (ARP chaos). |
| IP | Duplicate IPs = replies race; `ping` alternates between VMs. |
| rootfs file | Two kernels writing one ext4 journal = corruption. Always `cp` per VM. |

Convention that prevents 90% of bugs: **last MAC byte = last IP octet**
(`:02` ↔ `.2`, `:03` ↔ `.3`). Scripts enforce it; keep it when extending.

## 1. Files

- `commands.sh` — runbook (bridge → boot A/B/C → mesh test).
- `bridge-up.sh / bridge-down.sh` — create/destroy `fc-br0` (`10.0.0.1/24`),
  enable forwarding + NAT. Up is idempotent.
- `boot-vm.sh <A|B|C>` — full per-VM boot: pick TAP/IP/MAC/socket from the
  table, `cp` Lab 1 rootfs to `/tmp/fc-rootfs-<vm>.ext4`, create TAP,
  enslave to bridge (`ip link set ... master fc-br0`), API sequence,
  then SSH-configure guest net. Safe to run A/B/C in parallel (`&`).
- `mesh-test.sh` — graded matrix: every VM pings every other VM +
  gateway + Internet. Prints `PASS/FAIL` per pair + `bridge link`.

## 2. Quick start

```bash
cd lab-4-multiple-vms
bash bridge-up.sh
bash boot-vm.sh A & bash boot-vm.sh B & bash boot-vm.sh C &
wait
bash mesh-test.sh
```

Each `boot-vm.sh` copies the Lab 1 rootfs to `/tmp/fc-rootfs-<X>.ext4`
(copy-on-write workflow: never share one ext4 between VMs).

Watch it like a switch operator while it runs:

```bash
bridge link              # tap-a/b/c enslaved to fc-br0?
bridge fdb show br fc-br0  # learned MACs per port?
ip neigh                 # ARP: IP → MAC, REACHABLE?
ps aux | grep firecracker  # 3 VMM processes?
```

## 3. IP plan

| VM | TAP | Guest IP | MAC | Socket | Rootfs copy |
|---|---|---|---|---|---|
| A | tap-a | 10.0.0.2/24 | 06:00:AC:10:00:02 | /tmp/fc-a.socket | /tmp/fc-rootfs-a.ext4 |
| B | tap-b | 10.0.0.3/24 | 06:00:AC:10:00:03 | /tmp/fc-b.socket | /tmp/fc-rootfs-b.ext4 |
| C | tap-c | 10.0.0.4/24 | 06:00:AC:10:00:04 | /tmp/fc-c.socket | /tmp/fc-rootfs-c.ext4 |
| host br | fc-br0 | 10.0.0.1/24 | — | — | — |

## 4. Verify

- `bridge link` shows tap-a/b/c enslaved to fc-br0 (state `forwarding`).
- Every VM pings every other VM + 10.0.0.1 + 8.8.8.8 (`mesh-test.sh` all PASS).
- `ps aux | grep firecracker` shows 3 VMM processes (3 sockets, 3 logs).
- Bonus: `iperf3` between B and C (install once, run server in one guest,
  client in the other) — your first virtual-datacenter benchmark.

## Cleanup

```bash
for vm in a b c; do sudo curl -s -X PUT --unix-socket /tmp/fc-$vm.socket --data '{"action_type": "SendCtrlAltDel"}' http://localhost/actions || true; done
sleep 2; sudo pkill -f 'firecracker --api-sock /tmp/fc-' || true
bash bridge-down.sh
rm -f /tmp/fc-rootfs-*.ext4 /tmp/fc-?.socket /tmp/fc-?.log
```

Graceful first (`SendCtrlAltDel` ≈ guest `reboot`), `pkill` as backstop.

## Troubleshooting

- Two VMs same MAC/IP → ARP chaos: `ip neigh` flaps, pings alternate.
  Fix the table entry, reboot that VM only.
- Shared rootfs file → ext4 corruption (`dmesg` journal errors in guest).
  Always per-VM `cp`. (Production uses reflink/qcow2 overlays — Lab 7 notes.)
- Bridge has no IP → guests ping each other but not Internet; `bridge-up.sh`
  assigns 10.0.0.1 — check `ip addr show fc-br0`.
- Guest SSH hangs after boot → boot race: increase `sleep` before SSH in
  `boot-vm.sh` (slow disks need 4–5 s), or SSH manually to debug.
