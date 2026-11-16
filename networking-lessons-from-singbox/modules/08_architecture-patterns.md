# Module 8: Architecture Patterns, Glossary & Practice

This module covers reusable architecture patterns found in sing-box, a comprehensive glossary of networking terms, and hands-on configuration practice.

---

## Part 1: Design Patterns from the Codebase

### Pattern 1: Registry — Zero-Coupling Plugin System

Every protocol and transport registers itself at initialization. No central config file needed to discover them.

```python
# save as registry.py
class Registry:
    """Sing-box's registration pattern: new components announce themselves."""
    
    def __init__(self):
        self._registry = {}
    
    def register(self, type_name: str, creator_fn):
        """Register a component by type name"""
        if type_name in self._registry:
            print(f"  [WARNING] Overwriting registration for '{type_name}'")
        self._registry[type_name] = creator_fn
        print(f"  Registered: {type_name} → {creator_fn.__name__}")
    
    def create(self, type_name: str, **kwargs):
        """Create an instance by type name"""
        if type_name not in self._registry:
            raise KeyError(f"Unknown type: {type_name}. Available: {list(self._registry.keys())}")
        return self._registry[type_name](**kwargs)
    
    def list_types(self):
        return list(self._registry.keys())

# Usage example — simulating how sing-box registers protocols
registry = Registry()

# Each protocol file has its own init() that registers with the global registry
def new_direct_outbound(**kw):
    return {"type": "direct", "description": "Direct connection (no proxy)"}

def new_vmess_outbound(server, port, **kw):
    return {"type": "vmess", "server": server, "port": port}

def new_trojan_outbound(server, port, password, **kw):
    return {"type": "trojan", "server": server, "port": port, "password": password}

def new_socks_outbound(server, port, **kw):
    return {"type": "socks", "server": server, "port": port, "version": "5"}

# Registration (happens in each protocol's init.go or __init__.py)
registry.register("direct", new_direct_outbound)
registry.register("vmess", new_vmess_outbound)
registry.register("trojan", new_trojan_outbound)
registry.register("socks", new_socks_outbound)

# Creating outbounds — type name from config, no if/elif chain needed
config_types = ["direct", "vmess", "trojan", "socks"]
for t in config_types:
    if t == "direct":
        ob = registry.create(t)
    else:
        ob = registry.create(t, server="proxy.example.com", port=443)
    print(f"Created: {ob}")

print(f"\nAvailable types: {registry.list_types()}")
```

Run it:
```bash
python3 registry.py
```

**Why this matters:** When sing-box adds a new protocol (e.g., a new QUIC-based proxy), they don't need to modify the router or config parser. They just register the new type, and it works everywhere automatically. This is called **open-closed principle** — open for extension, closed for modification.

---

### Pattern 2: Adapter Interface — Pluggable Components

All inbounds and outbounds implement the same interfaces, allowing them to be mixed freely.

```python
# save as adapter.py
from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class ConnectionMeta:
    """Standard metadata passed between all components"""
    source_ip: str = ""
    source_port: int = 0
    destination_ip: str = ""
    destination_port: int = 0
    network: str = "tcp"
    domain: str = ""

class Outbound(ABC):
    """Every outbound implements this interface"""
    
    @abstractmethod
    def network(self) -> list[str]:
        """What networks this outbound supports"""
        pass
    
    @abstractmethod
    def dial(self, meta: ConnectionMeta):
        """Establish a connection to the destination"""
        pass
    
    @property
    @abstractmethod
    def tag(self) -> str:
        """Human-readable identifier"""
        pass

class DirectOutbound(Outbound):
    def network(self): return ["tcp", "udp"]
    def dial(self, meta):
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((meta.destination_ip, meta.destination_port))
        return f"Direct connection to {meta.domain or meta.destination_ip}"
    @property
    def tag(self): return "direct"

class SocksOutbound(Outbound):
    def __init__(self, server, port, version="5"):
        self.server = server
        self.port = port
        self._version = version
    
    def network(self): return ["tcp"]
    def dial(self, meta):
        return f"SOCKS{self._version} proxy {self.server}:{self.port} → {meta.domain or meta.destination_ip}"
    @property
    def tag(self): return f"socks-{self.server}"

class TLSOutbound(Outbound):
    """Wraps another outbound with TLS encryption"""
    
    def __init__(self, inner: Outbound):
        self.inner = inner
    
    def network(self): return self.inner.network()
    def dial(self, meta):
        return f"TLS wrapped: {self.inner.dial(meta)}"
    @property
    def tag(self): return f"tls({self.inner.tag})"

# Demo — mixing components freely
outbounds = [
    DirectOutbound(),
    SocksOutbound("1.2.3.4", 1080),
    TLSOutbound(SocksOutbound("5.6.7.8", 443)),
]

meta = ConnectionMeta(
    source_ip="192.168.1.10", destination_ip="93.184.216.34",
    destination_port=443, domain="example.com"
)

print("=== Pluggable Outbounds ===\n")
for ob in outbounds:
    result = ob.dial(meta)
    print(f"[{ob.tag:20}] → {result}")

print("\nAny outbound works with any inbound because they share interfaces.")
```

Run it:
```bash
python3 adapter.py
```

---

### Pattern 3: Suspended State Machine — Async DNS Resolution

The DNS rule engine suspends and resumes as it waits for DNS responses, rather than blocking.

```python
# save as dns_state_machine.py
import asyncio

class DNSStateMachine:
    """
    Instead of blocking while waiting for a DNS response,
    the state machine suspends and resumes when data arrives.
    
    This is how sing-box handles speculative DNS queries efficiently.
    """
    
    def __init__(self):
        self.futures = {}      # Named DNS results: name → resolved IP
        self.armed_rules = []  # Rules waiting for futures
        self.result_queue = []
    
    async def wait_for_dns(self, query_id: str, domain: str) -> str:
        """Simulate an async DNS lookup"""
        print(f"  [DNS] Querying {domain}... (async)")
        await asyncio.sleep(0.1)  # Simulate network delay
        result = f"93.184.{query_id}.{254 - query_id}"
        print(f"  [DNS] {domain} → {result}")
        return result
    
    async def evaluate_rule(self, rule_name: str, domain: str, cache_key: str | None = None) -> str | None:
        """Evaluate a DNS rule that may need to look up a future"""
        
        if cache_key and cache_key in self.futures:
            # Future already resolved — use cached result
            print(f"  [Rule {rule_name}] Cache hit: {cache_key} → {self.futures[cache_key]}")
            return self.futures[cache_key]
        
        if cache_key:
            # Not resolved yet — suspend and arm the rule
            print(f"  [Rule {rule_name}] Suspending for future '{cache_key}'...")
            self.armed_rules.append((rule_name, domain, cache_key))
            return None  # Will be resolved later
        
        # No dependency — resolve immediately
        result = await self.wait_for_dns(hash(domain) % 256, domain)
        print(f"  [Rule {rule_name}] Resolved: {result}")
        return result
    
    def resolve_future(self, name: str, ip: str):
        """A future resolved — notify all armed rules"""
        self.futures[name] = ip
        print(f"\n[FUTURE RESOLVED] {name} → {ip}")
        
        # Resume all rules waiting on this future
        remaining = []
        for rule_name, domain, cache_key in self.armed_rules:
            if cache_key == name:
                print(f"  [Resuming] Rule '{rule_name}' can now proceed")
            else:
                remaining.append((rule_name, domain, cache_key))
        
        self.armed_rules = remaining

# Demo the state machine
async def demo():
    sm = DNSStateMachine()
    
    print("=== Step 1: Evaluate rule for 'google.com' with named future ===")
    result = await sm.evaluate_rule("route-google", "google.com", cache_key="google_dns")
    print(f"  Result: {result}\n")
    
    print("=== Step 2: Evaluate rule that doesn't need a future ===")
    result = await sm.evaluate_rule("check-blocklist", "tracker.example.com")
    print(f"  Result: {result}\n")
    
    print("=== Step 3: Future resolves — previously suspended rules resume ===")
    await sm.resolve_future("google_dns", "142.250.80.46")

asyncio.run(demo())
```

Run it:
```bash
python3 dns_state_machine.py
```

---

### Pattern 4: Layered Wrapping (Russian Dolls)

Each layer adds its own header and processing, creating a chain of encapsulation.

```python
# save as layers.py
import struct, hashlib, base64, os

class PacketLayers:
    """Demonstrate how proxy data is wrapped in multiple protocol layers"""
    
    def __init__(self, original_data: bytes):
        self.original = original_data
        self.layers = []
    
    def add_vmess_layer(self):
        """Layer 1: VMess encrypted payload"""
        iv = os.urandom(16)
        header = b'V' + iv + struct.pack('!H', len(self.original))
        # Simplified: just show the structure, not actual encryption
        self.layers.append(("VMess", header + b'[ENCRYPTED]'))
        return len(header) + 9
    
    def add_websocket_frame(self):
        """Layer 2: WebSocket frame wrapping"""
        payload = self.layers[-1][1] if self.layers else self.original
        length = len(payload)
        
        # WebSocket frame: FIN+opcode | mask+length | [mask key if masked] | payload
        if length < 126:
            header = struct.pack('!BB', 0x81, 0x80 | length)  # FIN + TEXT + masked
        elif length < 65535:
            header = struct.pack('!BBH', 0x81, 0x80 | 126, length)
        
        self.layers.append(("WebSocket Frame", header + b'[FRAMED]'))
        return len(header)
    
    def add_tls_record(self):
        """Layer 3: TLS record wrapping"""
        content_type = 0x17  # Application data
        version = 0x0303     # TLS 1.3
        self.layers.append(("TLS Record", struct.pack('!BBH', content_type, version, 0)))
        return 5
    
    def add_tcp_header(self):
        """Layer 4: TCP header"""
        src_port = 54321
        dst_port = 443
        seq_num = 1000
        ack_num = 0
        data_offset = 5 << 4  # 20 bytes, no options
        flags = 0x18  # PSH + ACK
        
        self.layers.append(("TCP Header", struct.pack('!HHII BBH X', src_port, dst_port, seq_num, ack_num, data_offset, flags, 0)))
        return 20
    
    def show_structure(self):
        total = sum(len(l[1]) for l in self.layers) + len(self.original)
        
        print(f"{'Layer':<20} {'Size':>8} {'Cumulative':>12}")
        print("-" * 45)
        
        cumulative = 0
        # Show innermost layer first (what the user sends)
        for name, data in reversed(self.layers):
            size = len(data)
            cumulative += size
            print(f"{name:<20} {size:>8} {cumulative + len(self.original):>12}")
        
        # Original data at the core
        print(f"{'Original Data':<20} {len(self.original):>8} {total:>12}")
        print(f"{'TOTAL':<20} {'':>8} {total:>12}")
    
    def demonstrate_observer_view(self):
        """What different observers see at each layer"""
        print(f"\n=== Observer View (what each layer reveals) ===\n")
        
        observers = [
            ("Network scanner (packet capture)", "TCP to port 443 — looks like HTTPS"),
            ("TLS inspector", "Encrypted application data — can't see inside"),
            ("WebSocket analyzer", "TEXT frame with binary content"),
            ("VMess parser", "Version byte + IV + encrypted payload"),
            ("You (original sender)", f'"{self.original.decode()}"'),
        ]
        
        for observer, view in observers:
            print(f"  {observer:35} → {view}")

# Demo
packet = PacketLayers(b"GET / HTTP/1.1\r\nHost: example.com\r\n\r\n")
packet.add_vmess_layer()
packet.add_websocket_frame()
packet.add_tls_record()
packet.add_tcp_header()

print("=== Layer Structure ===\n")
packet.show_structure()
print("\nThe original data is nested deep inside 4 layers of protocol headers.")
packet.demonstrate_observer_view()
```

Run it:
```bash
python3 layers.py
```

---

## Part 2: Glossary

### Core Concepts

| Term | Definition |
|------|-----------|
| **Proxy** | An intermediary that forwards network requests on your behalf, masking your identity |
| **Outbound** | Your "exit point" — the proxy server or direct connection traffic uses to reach the internet |
| **Inbound** | How traffic enters your proxy system (TUN interface, SOCKS server, HTTP proxy server) |
| **Transport** | The wire-level encapsulation protocol (WebSocket, gRPC, QUIC) that wraps proxy payloads |
| **Protocol** | The rules governing client-server communication (VMess, Trojan, Shadowsocks, etc.) |
| **Rule** | A condition-action pair: "if X matches, do Y" |
| **FakeIP** | Synthetic IP addresses assigned to domains, enabling deferred DNS resolution |

### Networking Terms

| Term | Definition |
|------|-----------|
| **IP Address** | Unique identifier for a device on a network (e.g., 192.168.1.10) |
| **Port** | Application-level address on a device (e.g., port 80 = HTTP, port 443 = HTTPS) |
| **TCP** | Reliable, ordered byte stream with handshake and acknowledgment |
| **UDP** | Unreliable, connectionless datagram protocol — faster but no guarantees |
| **TLS** | Transport Layer Security — encryption for network connections (HTTPS's "S") |
| **DNS** | Domain Name System — translates human-readable domains to IP addresses |
| **DoH** | DNS over HTTPS — encrypted DNS queries sent as HTTPS requests |
| **DoT** | DNS over TLS — encrypted DNS queries sent over a dedicated TLS connection |
| **QUIC** | Quick UDP Internet Connections — Google's protocol replacing TCP+TLS, now standardized by IETF |
| **SNI** | Server Name Indication — plaintext domain name in TLS handshake (visible to observers) |
| **AEAD** | Authenticated Encryption with Associated Data — encryption + integrity verification |
| **NAT** | Network Address Translation — rewriting IP addresses in packets (used by FakeIP system) |

### sing-box Specific Terms

| Term | Definition |
|------|-----------|
| **Selector** | Manual outbound group — user picks which server to use |
| **URLTest** | Automatic outbound group — selects the lowest-latency healthy server |
| **Pre-match** | Phase where first packets are inspected before creating a full connection |
| **Rule Set** | A named collection of rules, loaded inline or from remote URL |
| **Detour** | Sending traffic through another outbound (proxy chaining) |
| **Stack** | The network implementation: system (OS kernel), gvisor (Go userspace), or mixed |
| **Sniff** | Protocol detection from first packets (TLS SNI, HTTP host, QUIC domain) |
| **RDRC** | Rejected Domain Response Cache — remembers rejected domains to avoid re-querying |

### Proxy Protocols Reference

| Protocol | Year | Transport | Key Feature |
|----------|------|-----------|-------------|
| **Shadowsocks** | 2012 | TCP/UDP, any transport | Simple stream cipher proxy; foundation for many later protocols |
| **VMess** | 2016 | Any (WS/gRPC/QUIC) | AEAD encryption + time-based replay protection |
| **VLESS** | 2019 | Any (WS/gRPC/QUIC) | Simplified VMess with flow control for XTLS |
| **Trojan** | 2018 | TCP/TLS only | Minimalist — prepends SHA3 password hash, payload is valid HTTP |
| **Hysteria v1** | 2021 | QUIC | "Bend" protocol for blocking circumvention; designed for lossy networks |
| **Hysteria v2** | 2023 | TCP + QUIC obfuscation | Stream-oriented, masquerades as HTTP/2 traffic |
| **TUIC** | 2022 | QUIC only | Native QUIC multiplexing — one connection for everything |
| **NaiveProxy** | 2017 | HTTPS | Request padding + encryption; designed specifically against DPI |
| **MASQUE** | 2022 | HTTP/HTTPS (RFC 9230) | IETF standard for multi-application proxy over HTTP |

---

## Part 3: Practice Configurations

### Scenario 1: Basic Setup with Bypass and Default Proxy

```jsonc
// config-basic.json
{
  "log": { "level": "info" },
  
  "inbounds": [{
    "type": "tun",
    "tag": "tun-in",
    "inet4_address": "172.19.0.1/30",
    "auto_route": true,
    "strict_route": true,
    "stack": "mixed"
  }],
  
  "outbounds": [
    { "type": "direct", "tag": "direct" },
    { "type": "block", "tag": "block" },
    { 
      "type": "vmess", 
      "tag": "proxy-us",
      "server": "us.example.com", 
      "server_port": 443,
      "uuid": "your-uuid-here",
      "security": "auto",
      "tls": { "enabled": true, "server_name": "us.example.com" },
      "transport": { "type": "ws", "path": "/ws" }
    }
  ],
  
  "route": {
    "rules": [
      // Bypass private networks (LAN, loopback, link-local)
      { "ip_is_private": true, "action": "direct" },
      
      // DNS traffic — handle locally
      { "protocol": "dns", "action": "hijack_dns" },
      
      // Default: all other traffic through US proxy
      { "action": "route", "outbound": "proxy-us" }
    ]
  }
}
```

### Scenario 2: Advanced Multi-Proxy with Per-Domain Routing

```jsonc
// config-advanced.json
{
  "dns": {
    "servers": [
      { "tag": "fakeip", "address": "fakeip" },
      { "tag": "google-doh", "address": "https://dns.google/resolve" },
      { "tag": "local", "address": "local" }
    ],
    "rules": [
      // Google domains resolve via Google DoH
      { "domain_keyword": ["google", "youtube"], "action": "route_server", "server": "google-doh" },
      
      // Chinese domains resolve via local DNS  
      { "domain_suffix": [".cn", ".com.cn"], "action": "route_server", "server": "local" },
      
      // Everything else gets FakeIP (deferred resolution)
      { "action": "route_server", "server": "fakeip" }
    ]
  },
  
  "outbounds": [
    { "type": "direct", "tag": "direct" },
    { 
      "type": "trojan", 
      "tag": "us-east",
      "server": "us1.example.com", "server_port": 443,
      "password": "your-password",
      "tls": { "enabled": true }
    },
    { 
      "type": "trojan", 
      "tag": "eu-central", 
      "server": "eu.example.com", "server_port": 443,
      "password": "your-password",
      "tls": { "enabled": true }
    }
  ],
  
  // URLTest group automatically picks fastest
  "route": {
    "rule_set": [
      {
        "tag": "social-media",
        "type": "inline",
        "rules": [
          { "domain": ["facebook.com", "twitter.com", "instagram.com", "tiktok.com"] }
        ]
      },
      {
        "tag": "streaming",
        "type": "inline", 
        "rules": [
          { "domain_keyword": ["netflix", "hulu", "disney", "spotify"] }
        ]
      },
      {
        "tag": "china-domain",
        "type": "inline",
        "rules": [
          { "domain_suffix": [".cn", ".com.cn", ".taobao.com", ".baidu.com"] }
        ]
      }
    ],
    
    "rules": [
      { "ip_is_private": true, "action": "direct" },
      { "protocol": "dns", "action": "hijack_dns" },
      
      // Social media → use US proxy (URLTest picks best)
      { "rule_set": ["social-media"], "action": "route", "outbound": "us-east" },
      
      // Streaming → EU proxy (better latency for video)
      { "rule_set": ["streaming"], "action": "route", "outbound": "eu-central" },
      
      // Chinese domains → direct (no need to proxy domestically)
      { "rule_set": ["china-domain"], "action": "direct" },
      
      // Everything else → US proxy, auto-select fastest
      { "action": "route", "outbound": "urltest-default" }
    ],
    
    "final": "direct"
  }
}
```

### Scenario 3: Proxy Chaining (Double Hop)

```jsonc
// config-chain.json — Route through two proxies for extra security
{
  "outbounds": [
    // First hop: public VPS (low cost, anywhere)
    { 
      "type": "vmess", 
      "tag": "hop1-public",
      "server": "public-vps.example.com", "server_port": 443,
      "uuid": "uuid-1",
      "tls": { "enabled": true }
    },
    
    // Second hop: private server (higher security)
    { 
      "type": "trojan", 
      "tag": "hop2-private",
      "server": "private.example.com", "server_port": 443,
      "password": "secret",
      "tls": { "enabled": true },
      
      // IMPORTANT: This outbound detours through hop1-public
      "detour": "hop1-public"
    }
  ],
  
  "route": {
    "rules": [
      { "action": "route", "outbound": "hop2-private" }
    ]
  }
}
```

Traffic flow: `You → hop2 (trojan) → hop1 (vmess) → Internet`. Both proxies must see your traffic as normal. Your last-hop proxy sees the IP of your first-hop proxy, not yours.

---

## Final Checkpoint: What You've Learned

| Module | Core Concept | Key Takeaway |
|--------|-------------|--------------|
| 0 | Internet fundamentals | TCP guarantees delivery; UDP is fast but unreliable |
| 1 | Proxy basics | A proxy forwards requests — HTTP reads content, SOCKS forwards blindly |
| 2 | Transport wrapping | Wrapping hides proxy traffic inside normal protocols (WS, gRPC) |
| 3 | Proxy protocols | Each protocol has different auth/encryption tradeoffs |
| 4 | Routing engine | Ordered rules, first match wins; pre-match optimizes before connection |
| 5 | DNS system | FakeIP defers resolution; DoH/DoT encrypt queries |
| 6 | TUN & VPN | Virtual network card captures all OS-level IP traffic |
| 7 | Complete journey | One request flows through: DNS → TUN → PreMatch → Rules → Group → Transport → Proxy |
| 8 | Patterns & practice | Registry, Adapter, State Machine, Layering — patterns you can reuse |

---

*End of course. You now understand how proxy platforms like sing-box work at every layer — from TCP packets to DNS rules to transport encapsulation.*
