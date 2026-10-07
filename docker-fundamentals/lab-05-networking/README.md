# Lab 5 — Networking: `veth`, `docker0`, and `-p 8080:80`

**Question:** the container has its own `eth0` and IP (e.g. `172.17.0.2`), yet `curl localhost:8080`
on the host reaches `nginx:80` inside. How does a packet get there?

**Answer:** a **net namespace** (private network stack) + a **veth pair** (virtual cable:
one end in container, one on host) plugged into the **`docker0` bridge** + **NAT/port-mapping**
(`iptables`/`nftables`) to the outside world.

## 0. Beginner primer

- **Network namespace**: a process group with its own interfaces, IPs, routes, firewall view.
  `sudo unshare --net bash` → `ip addr` shows only `lo` — total isolation.
- **veth pair**: two virtual NICs wired together. Packet in one pops out the other.
  Docker puts `eth0` inside, keeps the peer (`vethXXXX`) on the host.
- **Bridge (`docker0`)**: a virtual switch on the host. All default-network containers plug into it,
  so they can ping each other; the bridge also routes/NATs to your real NIC for Internet.
- **Port mapping `-p HOST:CTR`**: host listens on HOST port; kernel forwards (DNAT via
  `iptables -t nat` / nftables, plus a `docker-proxy` helper in some setups) into the container.
- **`--network host`**: skips all of this — container shares host net ns (fast, zero isolation).
- **`--network none`**: maximum isolation — only loopback.

```text
client → host:8080
            │  (NAT / docker-proxy)
            ▼
      docker0 bridge (172.17.0.1)
       /           \
  vethAAA       vethBBB
     │              │
 container web   container db
 172.17.0.2:80   172.17.0.3:5432
```

Outbound path is the reverse + MASQUERADE (source-NAT) so replies find their way back.

## 1. Run it

```bash
./demo.sh      # nginx on 8080, shows container IP, veth, bridge, curl test
./inspect.sh   # iptables/nft NAT rules, bridge members, container routes, two-container ping
./no-docker.sh # pure-Linux net ns: unshare --net, watch interfaces vanish
./cleanup.sh
```

## 2. Manual walkthrough

```bash
docker run -d --rm --name web -p 8080:80 nginx:alpine
docker inspect web --format 'IP={{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}} SandboxKey={{.NetworkSettings.SandboxKey}}'

# Inside: its own eth0, own routes
docker exec web ip addr
docker exec web ip route        # default via 172.17.0.1 dev eth0

# Host: bridge + veth peer appeared
ip link | grep -E 'docker0|veth'
ip addr show docker0
bridge link 2>/dev/null || brctl show 2>/dev/null || ip link show type bridge

# Which veth belongs to `web`? match ifindex ↔ eth0 peer:
CTR_PID=$(docker inspect web --format '{{.State.Pid}}')
sudo nsenter -t "$CTR_PID" -n ip link show eth0
ethtool -S <veth> 2>/dev/null | head  # optional

# Port path:
curl -s -o /dev/null -w '%{http_code}\n' localhost:8080
sudo iptables -t nat -L DOCKER -n --line-numbers 2>/dev/null | head -20 || sudo nft list chain ip nat DOCKER 2>/dev/null | head -30

# Container-to-container via bridge:
docker run -d --rm --name db2 nginx:alpine
docker exec web ping -c2 "$(docker inspect db2 --format '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}')"
docker rm -f db2

# Escape hatches:
docker run --rm --network none ubuntu:22.04 ip addr     # only lo
docker run --rm --network host ubuntu:22.04 ip addr     # SAME as host — no isolation!
```

## 3. No-Docker version

```bash
sudo unshare --net --fork bash
ip addr        # only lo — you are network-alone
ping 8.8.8.8  # fails: no veth, no bridge, no NAT
exit
```

Docker's job is wiring up everything that cell is missing.

## 4. Check your understanding

1. `docker exec web ip addr` vs host `ip addr` — why disjoint interface lists? (Separate net namespaces; only veth peers + bridge straddle the boundary.)
2. You publish `-p 8080:80` but `curl <container-IP>:80` from host also works — why? (Bridge IP is routable from host; `-p` additionally NATs host-port → container.)
3. When would you use `--network host`? (Low-latency/monitoring daemons that need host NICs; cost = container sees ALL host ports/interfaces.)

Next: **Lab 6** — cgroups: who enforces `--memory=100m --cpus=0.5`?
