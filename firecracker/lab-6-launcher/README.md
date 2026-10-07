# Lab 6 — Build a Tiny VM Launcher

> Beginner-friendly. You stop hand-typing `curl` and write the smallest
> program that *is* a microVM platform: configure → boot → use → destroy.

Moves you from "I can use Firecracker" to "I understand a microVM platform":

```text
create VM (spawn firecracker --api-sock)
   ↓
configure CPU + memory   (PUT /machine-config)
   ↓
configure kernel         (PUT /boot-source)
   ↓
configure rootfs         (PUT /drives/rootfs)
   ↓
configure network        (PUT /network-interfaces)
   ↓
start VM                 (PUT /actions InstanceStart)
```

## 0. Concepts (what the code is actually doing)

| Word | Meaning | In `launcher.py` |
|---|---|---|
| **Unix socket** | File-backed local connection (not TCP). Faster + no ports; permissions = file permissions. Firecracker listens on `/tmp/fc-*.socket`. | `UnixConn(http.client.HTTPConnection)` dials the file, then speaks plain HTTP. |
| **HTTP API** | Firecracker exposes resources (`/logger`, `/boot-source`, `/drives/rootfs`, …). `PUT` = "set this config", `GET /version` = health check, `PUT /actions` = commands. Order matters: configure everything *before* `InstanceStart`. | `put()` helper: `PUT` + fail loudly unless 200/201/204. |
| **`subprocess`** | Python spawning real programs (`ip tuntap`, `iptables`, `./firecracker`). Launcher shells out for net setup, keeps HTTP for VMM config. | `ensure_tap()` + `Popen([FC_BIN, --api-sock …])`. Must run with `sudo` (TAP/KVM need root). |
| **Idempotency** | Running twice = same result, not an error. `ip link del || true`, `iptables -C` before `-A`, `unlink socket` before bind. | Copy these patterns — every platform needs them. |
| **API-socket vs `--config-file`** | Same VM, two steering wheels. Socket = step-by-step (debuggable, scriptable). Config file = one JSON, VM auto-starts (simple, opaque). | `launcher.py` = socket mode; `vm.json.example` = file mode. Learn both, ship socket mode. |
| **`SendCtrlAltDel`** | Graceful guest reboot/shutdown via virtual keyboard. Guest OS syncs disks, then Firecracker exits. Opposite of `kill -9` (Lab 2's crash cord). | `launcher.py stop`. |

Read `launcher.py` in this order — it is deliberately short (~120 lines):
`argparse` (CLI) → `ensure_tap` (net) → `Popen firecracker` (spawn) →
`put(...)×6` (configure) → `InstanceStart` (go). `status`/`stop` are 3 lines each.

## 1. Files

- `commands.sh` — runbook (socket mode → file mode → full demo).
- `launcher.py` — stdlib-only launcher (no pip). Subcommands:
  `boot --kernel --rootfs [--socket --tap --guest-ip --mac --vcpu --mem]`,
  `status --socket`, `stop --socket`. Uses `http.client` over the Unix
  socket — no `curl` dependency.
- `vm.json.example` — same VM as one JSON file for `--config-file` mode.
  Edit the two absolute paths before use.
- `demo.sh` — end-to-end: globs Lab 1 assets, `boot`, `status`,
  SSH `uname -a + hello-from-guest`, prints the `stop` command.

## 2. Quick start

```bash
cd lab-6-launcher
python3 launcher.py --help
bash demo.sh
```

Manual (feel each stage):

```bash
sudo python3 launcher.py boot \
  --kernel ~/fc-lab1/vmlinux-* \
  --rootfs ~/fc-lab1/ubuntu-*.ext4 \
  --socket /tmp/fc-demo.socket \
  --tap tap-demo --guest-ip 172.16.0.2 --mac 06:00:AC:10:00:02

sudo python3 launcher.py status --socket /tmp/fc-demo.socket
ssh -i ~/fc-lab1/ubuntu-*.id_rsa root@172.16.0.2 'echo hello-from-guest'
sudo python3 launcher.py stop --socket /tmp/fc-demo.socket
```

Config-file mode (for comparison — no code, one shot):

```bash
cp vm.json.example /tmp/vm.json   # edit kernel/rootfs paths to absolute!
sudo ../lab-1-boot/firecracker --api-sock /tmp/fc-cfg.socket --config-file /tmp/vm.json
```

## 3. Verify (what each check proves)

- `boot` returns within ~2 s and `status` (`GET /version`) answers.
  Proves spawn + all 6 PUTs succeeded in order.
- `ssh root@172.16.0.2 'echo hello-from-guest'` works. Proves TAP+MAC+IP
  chain from Lab 3 was set up by *code*, not hand-typed commands.
- `stop` (`SendCtrlAltDel`) terminates the VMM process (`pgrep` empty).
  Proves graceful lifecycle — the platform cleans up after itself.

## Troubleshooting

- `Connection refused / No such file (socket)` → firecracker died or
  never spawned; run with `sudo` (TAP + `/dev/kvm` need root), check the
  `--log` file for bad kernel/rootfs paths.
- `iptables/HOST_IFACE` errors → VPN hijacked default route; export
  `HOST_IFACE` explicitly or disconnect VPN for the lab.
- Glob matches nothing (`vmlinux-*`) → run Lab 1 fetch first; launcher
  globs `~/fc-lab1/` by default (`FC_BIN` too — edit top of file if yours differs).
