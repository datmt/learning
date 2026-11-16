# Module 7: Complete Request Journey — End to End

This module traces a single web request through every layer of sing-box, from the moment you type a URL in your browser to the page loading on your screen. Every decision point, every transformation, every protocol exchange — all in one continuous flow.

---

## The Scenario

You configure sing-box with:
- TUN mode enabled (system-wide proxy)
- FakeIP DNS server active
- Multiple outbounds: `us-proxy`, `direct`, `block`
- Rules that route social media through US proxy, local traffic direct, everything else blocked

You type **`https://github.com`** in your browser. Let's trace every step.

---

## Phase 1: DNS Resolution (FakeIP)

```mermaid
sequenceDiagram
    participant B as Browser
    participant F as FakeIP Server<br/>(sing-box dns)
    participant S as sing-box Router<br/>(route/dns)
    
    Note over B: User types "https://github.com"
    
    B->>F: DNS query for "github.com"<br/>(UDP to 172.19.0.1:53)
    
    activate S
    S->>S: Check DNS rules<br/>Any match? No → use default
    
    S->>F: Route to FakeIP transport
    F->>F: Generate fake IP: 198.18.2.47
    F->>F: Store mapping:<br/>198.18.2.47 ↔ github.com
    F-->>S: Return fake IP
    S-->>B: "github.com = 198.18.2.47"
    deactivate S
    
    Note over B: Browser has an IP! (But it's fake)
```

**Key insight:** At no point did sing-box query a real DNS server for `github.com`. The FakeIP store now knows this domain, but won't resolve the real IP until actual traffic arrives.

---

## Phase 2: Connection Attempt

```mermaid
sequenceDiagram
    participant B as Browser<br/>(your app)
    participant K as OS Network Stack
    participant T as TUN Interface
    participant P as sing-box TUN Inbound
    
    Note over B: Browser connects to 198.18.2.47:443
    B->>K: TCP SYN → 198.18.2.47:443
    K->>T: Routing table says → tun0
    T->>P: Raw IP packet arrives
    
    activate P
    P->>P: "IP packet for 198.18.2.47"
    P->>F: FakeIP store lookup:<br/>198.18.2.47 → github.com
    F-->>P: Domain = "github.com" ✓
    
    Note over P: Pre-match phase begins
    P->>P: Read first packets (TCP handshake + TLS Client Hello)
    P->>P: Extract SNI: "github.com" from TLS
    
    P->>P: Enrich metadata:<br/>domain=github.com, process=firefox,<br/>src_ip=172.19.0.2
    deactivate P
```

---

## Phase 3: Rule Matching

```mermaid
sequenceDiagram
    participant R as Router<br/>evaluate rules
    participant M as Metadata:<br/>{domain:"github.com",<br/>port:443,<br/>process:"firefox"}
    
    activate R
    
    R->>M: Rule 1: source_ip=10.0.0.0/8?
    M-->>R: NO (src is 172.19.0.2)
    
    R->>M: Rule 2: destination_port=53?
    M-->>R: NO (port is 443)
    
    R->>M: Rule 3: domain ends with .cn?
    M-->>R: NO
    
    R->>M: Rule 4: domain = github.com → social media?
    M-->>R: NO
    
    R->>M: Rule 5: default rule (always match)
    M-->>R: YES ✓
    
    Note over R: Action: route_to = "us-proxy"
    
    deactivate R
```

---

## Phase 4: Outbound Resolution

```mermaid
sequenceDiagram
    participant O as Outbound<br/>Resolver
    participant G as Group:<br/>"us-proxy"<br/>(Selector)
    participant Members as Member outbounds:<br/>us-east (120ms)<br/>us-west (85ms)<br/>ca-central (210ms)
    
    O->>G: Resolve "us-proxy" group
    
    alt Selector (manual pick)
        G->>G: Return user's saved choice<br/>(e.g., "us-east")
    else URLTest (auto pick)
        G->>Members: Check all latencies
        Members-->>G: us-east=120ms, us-west=85ms, ca-central=210ms
        G->>G: Select fastest: us-west (85ms)
    end
    
    G-->>O: Resolved to "us-west"
    
    Note over O: Create outbound connection
```

---

## Phase 5: Transport Encapsulation

```mermaid
sequenceDiagram
    participant C as Client (sing-box)<br/>on your machine
    participant W as WebSocket Server<br/>on proxy server
    participant P as Proxy Backend<br/>(VMess/Trojan handler)
    
    Note over C: Prepare to send to us-west
    
    activate C
    
    C->>W: HTTP Upgrade Request<br/>GET /ws HTTP/1.1<br/>Upgrade: websocket<br/>Host: us-west.example.com
    W-->>C: 101 Switching Protocols
    
    Note over C,W: WebSocket connection established
    
    loop Proxy data exchange
        C->>W: [WS Frame: VMess payload encrypted]
        activate P
        W->>P: Extract VMess data, decrypt
        P->>P: Find destination: github.com:443
        P->>+W: Fetch from github.com
        W-->>-P: Response received
        P-->>W: Encrypted response
        W-->>C: [WS Frame: response payload]
        deactivate P
    end
    
    deactivate C
```

---

## Phase 6: Response Return and NAT Rewriting

```mermaid
sequenceDiagram
    participant G as github.com
    participant WS as WebSocket<br/>proxy server
    participant S as sing-box Router
    participant F as FakeIP NAT<br/>Rewriter
    participant T as TUN Interface
    participant B as Browser
    
    G->>WS: HTTP response: HTML, CSS, JS...
    WS->>S: Forward through proxy
    
    Note over S: Response arrives from us-west
    
    S->>F: "Response for connection to 198.18.2.47"
    F->>F: NAT rewrite:<br/>Change source IP<br/>from real GitHub IP<br/>to 198.18.2.47 (fake)
    
    F-->>S: Packet with rewritten source
    
    S->>T: Write response to TUN device
    T->>K: Kernel routes packet
    K->>B: "Response from 198.18.2.47"
    
    Note over B: Browser receives data,<br/>thinks it came directly<br/>from github.com's fake IP
    
    B->>B: Page renders ✓
```

---

## Complete Timeline — Every Step in Order

```python
# save as timeline.py — A summary of the complete request journey
steps = [
    ("0ms",  "DNS Query",     "Browser asks for github.com"),
    ("1ms",  "FakeIP Assign", "198.18.2.47 ← github.com (stored, no real DNS)"),
    ("1ms",  "TCP SYN",       "Browser sends SYN to 198.18.2.47:443"),
    ("2ms",  "TUN Intercept", "OS routes packet through tun0 → sing-box"),
    ("2ms",  "FakeIP Lookup", "198.18.2.47 → github.com (domain found)"),
    ("3ms",  "Pre-Match",     "Read first packets, extract TLS SNI = 'github.com'"),
    ("3ms",  "Metadata Build","{domain:'github.com', process:'firefox', port:443}"),
    ("4ms",  "Rule Match",    "Default rule → route to 'us-proxy' group"),
    ("4ms",  "Group Resolve", "URLTest picks us-west (85ms, fastest)"),
    ("5ms",  "WS Handshake",  "HTTP Upgrade → 101 Switching Protocols"),
    ("8ms",  "Proxy Connect", "VMess payload through WebSocket to proxy server"),
    ("93ms", "Proxy Fetch",   "Proxy server fetches github.com:443 (85ms round-trip)"),
    ("94ms", "NAT Rewrite",   "Response source IP → 198.18.2.47 (fakeIP)"),
    ("95ms", "TUN Inject",    "Response written to TUN device"),
    ("96ms", "Browser Read",  "Browser receives page content ✓"),
]

print("=== Complete Request Timeline: 'https://github.com' ===\n")
print(f"{'Time':>8} | {'Phase':<20} | Description")
print("-" * 75)
for time, phase, desc in steps:
    print(f"{time:>8} | {phase:<20} | {desc}")

total = int(steps[-1][0].replace('ms', ''))
print(f"\nTotal time from DNS to page load: ~{total}ms")
print(f"Real DNS resolution would add:  +20-50ms (depends on DNS server)")
print(f"FakeIP advantage: eliminates upfront DNS delay entirely")
```

Run it:
```bash
python3 timeline.py
```

---

## Variations: What Changes?

### Same Request, Different Protocol

| If the outbound uses... | Step 5 changes to... |
|------------------------|----------------------|
| **WebSocket** | HTTP Upgrade → WS frames (as shown above) |
| **gRPC** | gRPC stream setup instead of WS upgrade |
| **Plain TCP** | Direct TLS connection, no wrapping |
| **QUIC** | QUIC handshake (UDP-based), one-step |
| **Trojan** | No transport wrapping — just SHA3 key + HTTP |

### Same Request, Different Rule Match

```mermaid
graph LR
    A["github.com"] --> B["Default rule → proxy"]
    C["baidu.com.cn"] --> D["CN domain rule → local proxy"]
    E["192.168.1.100"] --> F["Local IP rule → direct"]
    G["ads.tracker.com"] --> H["Block rule → reject"]
    
    style A fill:#e1f5ff
    style C fill:#fff4e1
    style E fill:#69db7c
    style G fill:#ffcccc
```

Each destination takes a completely different path through the system, determined by which rule matches first.

### Different Transport for Same Proxy Server

```mermaid
graph TD
    A[Client] -->|WebSocket| B[Proxy: Looks like chat app]
    A -->|gRPC| C[Proxy: Looks like Google API]
    A -->|HTTP Upgrade| D[Proxy: Looks like file download]
    
    style B fill:#ccffcc
    style C fill:#ccffcc
    style D fill:#ccffcc
```

The proxy server backend is identical — same VMess/Trojan protocol. Only the transport layer (how data gets to the server) changes. This is why sing-box separates protocols from transports: **any protocol can use any transport**.

---

## Reverse Journey: The Response

The response follows the reverse path, but with one key transformation:

```mermaid
sequenceDiagram
    participant P as Proxy server<br/>delivers real data
    participant S as Router
    participant N as NAT Engine<br/>(FakeIP rewrite)
    participant T as TUN
    participant B as Browser
    
    Note over P,B:"The browser sees the response as coming from 198.18.2.47<br/>Not from the real GitHub IP (e.g., 140.82.121.3)"
    
    P->>S: Real data from github.com
    S->>N: NAT rewrite required
    N->>N: Rewrite source IP:<br/>140.82.121.3 → 198.18.2.47
    N->>T: Inject rewritten packet
    T->>B: Browser reads response
```

The NAT rewrite is what makes FakeIP seamless for the browser. The browser never learns that `github.com`'s real IP address — it only sees addresses from the FakeIP range, which belong to sing-box's internal network.

---

## What If Something Goes Wrong?

### Error Scenarios

| Failure Point | What Happens | How sing-box Handles It |
|--------------|-------------|------------------------|
| **FakeIP store full** | Can't assign new IPs | Falls back to real DNS resolution |
| **Rule match fails** | No matching rule | Uses default outbound (or blocks) |
| **Proxy server unreachable** | Connection timeout | URLTest switches to next best outbound |
| **WS server rejects** | 404/502 from proxy | Log error, try fallback outbound |
| **TLS handshake fails** | Certificate verification error | Depending on config: reject or skip verify |

### Timeout and Retry Flow

```mermaid
sequenceDiagram
    participant C as Client
    participant R as Router
    participant P1 as Proxy A<br/>(first attempt)
    participant P2 as Proxy B<br/>(fallback)
    
    C->>R: Request
    R->>P1: Connect to proxy A
    
    alt Success (< 3s timeout)
        P1-->>C: Response ✓
    else Timeout or error
        P1--x R: Failed
        R->>P2: Try proxy B (URLTest fallback)
        P2-->>C: Response ✓
    end
```

sing-box's URLTest group automatically tracks failures and avoids unhealthy outbounds for a configurable cooldown period.

---

## Exercises

### Exercise 1: Build a Complete Request Tracer

```python
# save as request_tracer.py — Track every phase of a request
import time
from dataclasses import dataclass, field
from typing import List

@dataclass
class TraceEntry:
    time: float
    phase: str
    detail: str
    duration: float = 0.0

class RequestTracer:
    def __init__(self):
        self.entries: List[TraceEntry] = []
        self._start = None
    
    def start(self):
        self._start = time.time()
    
    def step(self, phase: str, detail: str, duration: float = 0.0):
        elapsed = (time.time() - self._start) * 1000
        self.entries.append(TraceEntry(elapsed, phase, detail, duration))
    
    def print_trace(self):
        print(f"\n{'Elapsed':>10} | {'Phase':<25} | {'Detail'}")
        print("-" * 70)
        for e in self.entries:
            dur_str = f"+{e.duration*1000:.1f}ms" if e.duration > 0 else ""
            print(f"{e.time:>8.2f}ms | {e.phase:<25} | {e.detail} {dur_str}")

tracer = RequestTracer()
tracer.start()

# Simulate each phase of the request
tracer.step("DNS", "FakeIP assignment", 0.001)
tracer.step("DNS", "Stored: 198.18.2.47 → github.com")

tracer.step("TCP", "Browser → TUN interface", 0.002)
tracer.step("TUN", "Packet received by sing-box")

tracer.step("PreMatch", "SNI extraction: 'github.com'", 0.003)
tracer.step("PreMatch", "Metadata enriched (process, src_ip)")

tracer.step("Routing", "Rule 5 (default) matched")
tracer.step("Routing", "Group 'us-proxy' → URLTest selects us-west")

tracer.step("Transport", "WebSocket upgrade request sent")
tracer.step("Transport", "101 Switching Protocols received")

tracer.step("Proxy", "VMess payload encapsulated and sent")
tracer.step("Proxy", "Response received from proxy server", 0.085)

tracer.step("NAT", "Source IP rewritten: 140.82.121.3 → 198.18.2.47")
tracer.step("TUN", "Response injected into TUN device")
tracer.step("TCP", "Browser receives page data ✓")

tracer.print_trace()
```

Run it:
```bash
python3 request_tracer.py
```

---

*Next: [Module 8: Architecture Patterns](08_architecture-patterns.md) — Reusable design patterns from sing-box's codebase, glossary, and practice configurations.*
