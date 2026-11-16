# Proxy Networking Course

A practical, hands-on guide to understanding how proxy platforms like **sing-box** work — from TCP packets to rule-based routing to protocol encapsulation.

Each module is self-contained and includes runnable code examples. Read them in order or jump to topics that interest you.

---

## Prerequisites

- **Python 3.10+** — For all code examples
- **curl, dig** — For DNS and HTTP commands
- **socat or ncat** — For TCP testing
- **Linux/macOS** — Most commands work on Linux; macOS may need minor adjustments
- **Optional: root/sudo** — For TUN interface exercises

### Install Dependencies

```bash
# Python packages used in examples
pip3 install cryptography

# DNS tools (usually pre-installed)
sudo apt install dnsutils socat  # Debian/Ubuntu
sudo dnf install bind-utils socat  # Fedora/RHEL
brew install bind socat  # macOS

# HTTP testing (usually pre-installed)
which curl && echo "curl available"
```

---

## Course Outline

| Module | Topic | Difficulty | What You'll Build |
|--------|-------|-----------|-------------------|
| [00](modules/00_internet-fundamentals.md) | Internet Fundamentals | Beginner | TCP server, HTTP server, TLS inspection |
| [01](modules/01_proxy-basics.md) | What Is a Proxy | Beginner | HTTP proxy, SOCKS5 proxy, chain proxy |
| [02](modules/02_transport-wrapping.md) | Transport Wrapping | Intermediate | WebSocket server/client, frame analyzer |
| [03](modules/03_proxy-protocols.md) | Proxy Protocols | Intermediate | AEAD encryption, Trojan auth, VMess packets |
| [04](modules/04_routing-engine.md) | Routing Engine | Intermediate | Rule-based router, selector group, URLTest |
| [05](modules/05_dns-system.md) | DNS System | Intermediate | FakeIP store, DNS cache, rule engine |
| [06](modules/06_tun-vpn.md) | TUN & VPN | Advanced | Virtual interface, packet injection simulation |
| [07](modules/07_complete-journey.md) | Complete Journey | Review | End-to-end request tracer |
| [08](modules/08_architecture-patterns.md) | Patterns & Practice | Review | Registry pattern, adapter system, configs |

---

## Quick Start — First 15 Minutes

If you're new to networking, start here:

```bash
# Step 1: Understand TCP
python3 course/modules/00_internet-fundamentals.md
# Run: socat TCP-LISTEN:9999,fork SYSTEM:'cat' (server) + ncat localhost 9999 (client)

# Step 2: Build your first proxy
cp course/modules/01_proxy-basics.py basic_http_proxy.py
python3 basic_http_proxy.py &
curl -x http://127.0.0.1:8080 http://httpbin.org/ip

# Step 3: See DNS in action
dig example.com +short
dig @8.8.8.8 example.com +trace | head -10

# Step 4: Read Module 0 (Internet Fundamentals) — it covers all of the above in detail
```

---

## How This Course Relates to sing-box

Every module maps to real parts of the sing-box codebase:

| Your Code | sing-box Equivalent |
|-----------|-------------------|
| `basic_http_proxy.py` | `protocol/http/inbound.go` + `protocol/http/outbound.go` |
| `ws_server.py` / `ws_client.py` | `transport/v2raywebsocket/server.go` / `client.go` |
| `aes_gcm_encrypt()` in module 3 | `protocol/shadowsocks/outbound.go` (AEAD ciphers) |
| `router.py` | `route/route.go` + `route/router.go` |
| `fakeip.py` | `dns/transport/fakeip/fakeip.go` |
| `tun_sim.py` | `protocol/tun/inbound.go` + `transport/device/stack.go` |

The code examples use Python for accessibility, but the concepts are identical to the Go implementation in sing-box. When you understand a concept here, you can read the corresponding sing-box file and recognize the same pattern immediately.

---

## Exercise Solutions

All example scripts are embedded in their respective module files (marked as `save as X.py`). Copy them into your working directory and run with `python3 filename.py`.

To verify everything works:
```bash
# Each module has a "Run it:" command at the bottom of its exercises section
# Just follow along, step by step.

# Quick sanity check — does Python work?
python3 -c "print('Python OK')"

# Does networking work?
curl -s ifconfig.me && echo " - Internet reachable"

# Does DNS work?
dig +short example.com
```

---

## Contributing

Found a typo, want to add an exercise, or have a better explanation for a concept? Open a pull request.

---

## License

Same license as sing-box (MIT). The educational content here is free to use, adapt, and share.
