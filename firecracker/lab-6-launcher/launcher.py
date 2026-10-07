#!/usr/bin/env python3
"""Tiny Firecracker launcher — stdlib only, talks to --api-sock via Unix socket.

Usage:
  sudo python3 launcher.py boot --kernel K --rootfs R [--socket S] [--tap T]
  sudo python3 launcher.py status --socket S
  sudo python3 launcher.py stop --socket S
"""
import argparse, glob, json, os, socket, subprocess, sys, time, http.client

FC_BIN = os.path.expanduser("~/fc-lab1/firecracker")
if not os.path.exists(FC_BIN):
    FC_BIN = "./firecracker"  # fallback: lab-1 binary next to labs


class UnixConn(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost")
        self._path = path
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self._path)


def api(sock_path, method, url, body=None):
    c = UnixConn(sock_path)
    data = json.dumps(body).encode() if body is not None else None
    c.request(method, url, body=data, headers={"Content-Type": "application/json"})
    r = c.getresponse()
    payload = r.read().decode(errors="replace")
    return r.status, payload


def put(sock, url, body):
    st, out = api(sock, "PUT", url, body)
    if st not in (200, 201, 204):
        print(f"PUT {url} -> {st}: {out}", file=sys.stderr)
        sys.exit(1)
    print(f"ok {url}")


def ensure_tap(tap, tap_ip="172.16.0.1/30"):
    subprocess.run(["ip", "link", "del", tap], capture_output=True)
    subprocess.run(["ip", "tuntap", "add", "dev", tap, "mode", "tap"], check=True)
    subprocess.run(["ip", "addr", "add", tap_ip, "dev", tap], check=True)
    subprocess.run(["ip", "link", "set", "dev", tap, "up"], check=True)
    subprocess.run(["sh", "-c", "echo 1 > /proc/sys/net/ipv4/ip_forward"], check=True)
    subprocess.run(["iptables", "-P", "FORWARD", "ACCEPT"], check=True)
    host_if = subprocess.run(["sh", "-c", "ip -j route list default | jq -r '.[0].dev'"],
                             capture_output=True, text=True).stdout.strip()
    if host_if:
        nat = subprocess.run(["iptables", "-t", "nat", "-C", "POSTROUTING",
                              "-o", host_if, "-j", "MASQUERADE"])
        if nat.returncode != 0:
            subprocess.run(["iptables", "-t", "nat", "-A", "POSTROUTING",
                            "-o", host_if, "-j", "MASQUERADE"], check=True)


def cmd_boot(a):
    kernel = glob.glob(os.path.expanduser(a.kernel))[0]
    rootfs = glob.glob(os.path.expanduser(a.rootfs))[0]
    boot_args = "console=ttyS0 reboot=k panic=1"
    if os.uname().machine == "aarch64":
        boot_args = "keep_bootcon " + boot_args
    try:
        os.unlink(a.socket)
    except FileNotFoundError:
        pass
    ensure_tap(a.tap)
    proc = subprocess.Popen(["sudo", FC_BIN, "--api-sock", a.socket, "--enable-pci"]
                            if os.geteuid() != 0 else
                            [FC_BIN, "--api-sock", a.socket, "--enable-pci"])
    for _ in range(50):
        if os.path.exists(a.socket):
            break
        time.sleep(0.1)
    put(a.socket, "/logger", {"log_path": a.log, "level": "Info",
                              "show_level": True, "show_log_origin": True})
    put(a.socket, "/boot-source", {"kernel_image_path": kernel, "boot_args": boot_args})
    put(a.socket, "/machine-config", {"vcpu_count": a.vcpu, "mem_size_mib": a.mem})
    put(a.socket, "/drives/rootfs", {"drive_id": "rootfs", "path_on_host": rootfs,
                                     "is_root_device": True, "is_read_only": False})
    put(a.socket, f"/network-interfaces/net1",
        {"iface_id": "net1", "guest_mac": a.mac, "host_dev_name": a.tap})
    time.sleep(0.2)
    put(a.socket, "/actions", {"action_type": "InstanceStart"})
    print(f"VM running (vmm pid guess via pgrep). socket={a.socket} tap={a.tap}")
    print(f"guest net: ssh root@{a.guest_ip} (set IP via guest-net.sh or cloud-init)")


def cmd_status(a):
    st, out = api(a.socket, "GET", "/version")
    print(f"GET /version -> {st}: {out}")


def cmd_stop(a):
    st, out = api(a.socket, "PUT", "/actions", {"action_type": "SendCtrlAltDel"})
    print(f"stop -> {st}: {out}")


p = argparse.ArgumentParser(description="Tiny Firecracker launcher")
sub = p.add_subparsers(dest="cmd", required=True)
b = sub.add_parser("boot"); b.add_argument("--kernel", required=True)
b.add_argument("--rootfs", required=True); b.add_argument("--socket", default="/tmp/fc-demo.socket")
b.add_argument("--tap", default="tap-demo"); b.add_argument("--guest-ip", default="172.16.0.2")
b.add_argument("--mac", default="06:00:AC:10:00:02"); b.add_argument("--vcpu", type=int, default=1)
b.add_argument("--mem", type=int, default=256); b.add_argument("--log", default="/tmp/fc-demo.log")
s = sub.add_parser("status"); s.add_argument("--socket", default="/tmp/fc-demo.socket")
t = sub.add_parser("stop"); t.add_argument("--socket", default="/tmp/fc-demo.socket")
args = p.parse_args()
{"boot": cmd_boot, "status": cmd_status, "stop": cmd_stop}[args.cmd](args)
