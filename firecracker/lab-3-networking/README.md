# Lab 3 — Networking (TAP + NAT + Routing)

> Beginner-friendly. This is the core network-security lab: by the end
> you can explain every hop of `microVM → host → Internet` and inspect
> each one.

Build:

```text
             Host
              │
        ┌─────┴─────┐
        │   tap0    │  172.16.0.1/30  (host end of cable)
        └─────┬─────┘
              │ virtio-net (Firecracker shuttles Ethernet frames)
        ┌─────▼─────┐
        │ Firecracker│  172.16.0.2, MAC 06:00:AC:10:00:02 (guest end: eth0)
        │   microVM  │
        └─────────────┘
```

Then: `microVM → host (router+NAT) → Internet`.

## 0. Networking primer (just enough to survive this lab)

Data travels in layers. You only need two today:

- **L2 — Ethernet / MAC.** Physical-ish addressing on one cable/segment.
  A *frame* says `from MAC_A to MAC_B`. Switches/TAPs forward frames.
  No routing, no Internet — just "who's on this wire?"
- **L3 — IP / routing.** Logical addressing across networks. A *packet*
  says `from 172.16.0.2 to 8.8.8.8`. *Routers* forward packets between
  networks using a *routing table* (`ip route`).

Supporting cast:

| Word | Meaning | In this lab |
|---|---|---|
| **TAP** | Virtual Ethernet cable: software interface that sends/receives L2 frames. Host end `tap0`, guest end `eth0`. | The only wire. `ping` = frames across it. |
| **virtio-net** | Fast paravirtual NIC (no fake Realtek). Guest driver puts frames in a shared ring; Firecracker copies them to `tap0`. | Why throughput is decent despite tiny VMM. |
| **IP + CIDR (`/30`)** | Address + how many bits are "network" vs "host". `/30` = 4 addresses total, 2 usable. `/24` (Lab 4) = 256 addresses. | `.0` net, `.1` host, `.2` guest, `.3` broadcast. |
| **Subnet** | The "same wire" group. Two IPs in the same subnet talk directly; otherwise via a router. | `172.16.0.1` and `.2` share `/30` → direct. |
| **Default route** | "Send everything unknown to THIS router." Guest: `via 172.16.0.1`. Host: `via <your-wifi-router>`. | Without it, guest can ping host but not Internet. |
| **IP forwarding** | Host acting as a router: `/proc/sys/net/ipv4/ip_forward=1`. Off by default on workstations. | The one `echo 1 > ...` that makes egress possible. |
| **NAT / MASQUERADE** | Rewrite private source IP (`172.16.0.2`) to host's public IP on the way out; un-rewrite replies on the way back. Needed because the Internet can't route your private IP. | One `iptables -t nat` rule. Check with `iptables -t nat -L`. |
| **ARP** | "Who has IP X? Tell me your MAC." L2↔L3 glue. `ip neigh` shows the cache (`REACHABLE` = healthy). | Guest ARPs for `172.16.0.1`; MAC mismatch = blackhole. |
| **DNS (`/etc/resolv.conf`)** | Names → IPs (`example.com` → `93.184.216.34`). Ping-by-IP can work while curl-by-name fails = DNS bug, not routing bug. | Guest needs `nameserver 8.8.8.8` written by hand. |
| **iptables vs nftables** | Firewall/NAT frameworks. Arch uses the `nf_tables` backend; the `iptables` CLI is translated automatically. Rules you add with `iptables` show up in `nft list ruleset`. | Scripts use `iptables` (portable); either CLI is fine. |

Packet walk (guest curls `https://example.com`):

```text
1. guest app → getaddrinfo → DNS query to 8.8.8.8 (via default route → eth0)
2. guest kernel wraps in Ethernet frame (src MAC :02 → host TAP MAC) → virtio ring
3. Firecracker copies frame to tap0 → host kernel receives it
4. host routes: dst 8.8.8.8 ≠ local → forward out HOST_IFACE (wifi/eth)
5. NAT rewrites src 172.16.0.2 → <host-ip>, remembers mapping (conntrack)
6. reply returns → NAT un-rewrites → host routes to tap0 → frame to guest
```

## 1. Files

- `commands.sh` — runbook (up → boot → guest-net → verify).
- `host-net.sh [up|down]` — creates `tap0`, assigns `172.16.0.1/30`,
  enables forwarding, installs MASQUERADE on auto-detected `HOST_IFACE`.
  Idempotent (`-C` check before `-A`).
- `guest-net.sh` — run **inside** guest (or pipe over SSH): assigns
  `172.16.0.2/30` to `eth0`, adds default route, writes DNS (+ the
  `single-request-reopen` workaround for IPv6 NAT stalls).
- `verify-net.sh` — graded matrix: TAP exists, forwarding on, NAT present,
  host→guest ping, guest→host ping, guest→Internet ping, guest DNS+curl,
  then prints ARP + TAP counters.

## 2. Quick start

```bash
cd lab-3-networking
bash host-net.sh up
# boot VM (21-boot-ssh.sh already does host-net inline,
# or boot serial VM + attach network before InstanceStart)
bash verify-net.sh
```

Manual guest setup (if you serial-booted and want to feel each step):

```bash
# inside guest:
ip addr add 172.16.0.2/30 dev eth0
ip link set eth0 up
ip route add default via 172.16.0.1 dev eth0
echo 'nameserver 8.8.8.8' > /etc/resolv.conf
ping -c2 172.16.0.1   # L3 to host — should work first
ping -c2 8.8.8.8      # Internet by IP — needs forwarding+NAT
curl -sI https://example.com  # needs DNS too
```

## 3. IP plan

| Entity | IP | MAC | Role |
|---|---|---|---|
| host tap0 | 172.16.0.1/30 | host-owned | guest's default gateway + NAT router |
| guest eth0 | 172.16.0.2 | 06:00:AC:10:00:02 | last MAC byte `02` ↔ IP `.2` (keep in sync!) |

`/30` = 4 addresses, 2 usable. Deliberately tiny — Lab 4 upgrades to a bridge.

## 4. Verify

```bash
bash verify-net.sh
# guest → host: ping 172.16.0.1 OK      (L3 over TAP)
# guest → internet: ping 8.8.8.8 OK     (forwarding + NAT)
# guest → name: curl example.com 200    (DNS + egress)
# host → guest: ssh root@172.16.0.2 OK  (reverse path)
# ip neigh shows REACHABLE, iptables -t nat -L shows MASQUERADE
```

Learn to tell failures apart: IP-works-but-DNS-fails = `resolv.conf`;
host-works-but-Internet-fails = forwarding/NAT; nothing-works = TAP down
or MAC/IP mismatch (`ip neigh` shows `FAILED`).

## Cleanup

```bash
bash host-net.sh down
```

(NAT rule stays — harmless. TAP deletion is what matters.)

## Troubleshooting

- Guest→host OK, Internet fails → forwarding/NAT missing. Re-run `host-net.sh up`; verify `cat /proc/sys/net/ipv4/ip_forward` = `1`.
- `HOST_IFACE` wrong (VPN/WLAN active?) → `HOST_IFACE=wlan0 bash host-net.sh up`. The auto-detect picks the *default route* iface; VPNs hijack it.
- DNS fails but `ping 8.8.8.8` works → guest `/etc/resolv.conf` missing; run `guest-net.sh`.
- `iptables: Chain already exists` → harmless; scripts use `-C` checks.
- Slow DNS, fast IPs → IPv6 AAAA stall behind NAT; ensure guest resolv.conf has `options single-request-reopen`.
