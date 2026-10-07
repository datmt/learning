#!/usr/bin/env python3
"""Mini serverless on Firecracker: POST /run -> fresh microVM per job. Stdlib only. Must run as root."""
import argparse, glob, json, os, shutil, socket, subprocess, threading, time, http.client, http.server

A = None
LOCK = threading.Lock()
JOBN = 0
STATS = {"jobs_total": 0, "total_ms": 0}
BR = "fc-br0"
BASE = os.path.expanduser("~/fc-lab1")
KERNEL = sorted(glob.glob(os.path.join(BASE, "vmlinux-*")))
ROOTFS = sorted(glob.glob(os.path.join(BASE, "ubuntu-*.ext4")))
KEYS = sorted([k for k in glob.glob(os.path.join(BASE, "ubuntu-*.id_rsa")) if not k.endswith(".pub")])
FC_BIN = os.path.join(BASE, "firecracker")


class UnixConn(http.client.HTTPConnection):
    def __init__(self, path):
        super().__init__("localhost"); self._p = path
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(self._p)


def api(sock, method, url, body=None):
    c = UnixConn(sock)
    data = json.dumps(body).encode() if body is not None else None
    c.request(method, url, body=data, headers={"Content-Type": "application/json"})
    r = c.getresponse(); return r.status, r.read().decode(errors="replace")


def put(sock, url, body):
    st, out = api(sock, "PUT", url, body)
    if st not in (200, 201, 204):
        raise RuntimeError(f"PUT {url} -> {st}: {out}")


def sh(*cmd, check=True):
    return subprocess.run(cmd, check=check, capture_output=True, text=True)


def ensure_bridge():
    sh("ip", "link", "add", "name", BR, "type", "bridge", check=False)
    sh("ip", "addr", "add", "10.0.0.1/24", "dev", BR, check=False)
    sh("ip", "link", "set", "dev", BR, "up")
    sh("sh", "-c", "echo 1 > /proc/sys/net/ipv4/ip_forward")
    sh("iptables", "-P", "FORWARD", "ACCEPT")
    host_if = sh("sh", "-c", "ip -j route list default | jq -r '.[0].dev'").stdout.strip()
    if host_if and sh("iptables", "-t", "nat", "-C", "POSTROUTING",
                      "-o", host_if, "-j", "MASQUERADE", check=False).returncode != 0:
        sh("iptables", "-t", "nat", "-A", "POSTROUTING", "-o", host_if, "-j", "MASQUERADE")


def boot_vm(jobid, ip_last):
    tap = f"tap-j{jobid}"
    sock = f"/tmp/fc-job-{jobid}.socket"
    log = f"/tmp/fc-job-{jobid}.log"
    rootfs = f"/tmp/fc-job-{jobid}.ext4"
    ip = f"10.0.0.{ip_last}"
    mac = f"06:00:AC:10:00:{ip_last:02X}"
    shutil.copyfile(ROOTFS[0], rootfs)
    try: os.unlink(sock)
    except FileNotFoundError: pass
    sh("ip", "link", "del", tap, check=False)
    sh("ip", "tuntap", "add", "dev", tap, "mode", "tap")
    sh("ip", "link", "set", "dev", tap, "up")
    sh("ip", "link", "set", "dev", tap, "master", BR)
    boot_args = "console=ttyS0 reboot=k panic=1"
    if os.uname().machine == "aarch64": boot_args = "keep_bootcon " + boot_args
    subprocess.Popen([FC_BIN, "--api-sock", sock, "--enable-pci"],
                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    for _ in range(50):
        if os.path.exists(sock): break
        time.sleep(0.1)
    put(sock, "/logger", {"log_path": log, "level": "Warning", "show_level": True, "show_log_origin": True})
    put(sock, "/boot-source", {"kernel_image_path": KERNEL[0], "boot_args": boot_args})
    put(sock, "/machine-config", {"vcpu_count": 1, "mem_size_mib": 256})
    put(sock, "/drives/rootfs", {"drive_id": "rootfs", "path_on_host": rootfs,
                                 "is_root_device": True, "is_read_only": False})
    put(sock, "/network-interfaces/net1", {"iface_id": "net1", "guest_mac": mac, "host_dev_name": tap})
    time.sleep(0.2)
    put(sock, "/actions", {"action_type": "InstanceStart"})
    return {"tap": tap, "sock": sock, "rootfs": rootfs, "ip": ip, "mac": mac}


def destroy_vm(vm):
    try: api(vm["sock"], "PUT", "/actions", {"action_type": "SendCtrlAltDel"})
    except Exception: pass
    time.sleep(1)
    sh("ip", "link", "del", vm["tap"], check=False)
    for f in (vm["sock"], vm["rootfs"]):
        try: os.unlink(f)
        except FileNotFoundError: pass


def run_in_vm(vm, cmd, timeout=20):
    # wait for SSH, then exec
    deadline = time.time() + A.boot_wait
    while time.time() < deadline:
        r = subprocess.run(["ssh", "-i", KEYS[0], "-o", "StrictHostKeyChecking=no",
                            "-o", "ConnectTimeout=2", f"root@{vm['ip']}",
                            "ip addr add %(ip)s/24 dev eth0 2>/dev/null; ip link set eth0 up; ip route add default via 10.0.0.1 dev eth0 2>/dev/null; echo ok" % {"ip": vm["ip"]}],
                           capture_output=True)
        if r.returncode == 0: break
        time.sleep(0.5)
    r = subprocess.run(["ssh", "-i", KEYS[0], "-o", "StrictHostKeyChecking=no",
                        "-o", "ConnectTimeout=5", f"root@{vm['ip']}", cmd],
                       capture_output=True, text=True, timeout=timeout)
    return (r.stdout + r.stderr).strip()


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a): pass
    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path == "/health": return self._json(200, {"ok": True})
        if self.path == "/stats":
            with LOCK: s = dict(STATS)
            avg = s["total_ms"] / s["jobs_total"] if s["jobs_total"] else 0
            return self._json(200, {**s, "avg_ms": round(avg, 1)})
        return self._json(404, {"error": "use POST /run"})
    def do_POST(self):
        if self.path != "/run": return self._json(404, {"error": "use POST /run"})
        n = int(self.headers.get("Content-Length", 0))
        try: cmd = json.loads(self.rfile.read(n) or b"{}").get("cmd", "echo hello-from-vm; uname -a")
        except Exception: return self._json(400, {"error": "bad JSON"})
        with LOCK:
            global JOBN; JOBN += 1; jid = JOBN
        ip_last = 10 + (jid % 200)
        t0 = time.time()
        vm = None
        try:
            vm = boot_vm(jid, ip_last)
            out = run_in_vm(vm, cmd)
            ms = round((time.time() - t0) * 1000)
            with LOCK:
                STATS["jobs_total"] += 1; STATS["total_ms"] += ms
            return self._json(200, {"vm": f"job-{jid}", "ms": ms, "output": out})
        except Exception as e:
            return self._json(500, {"error": str(e)})
        finally:
            if vm: destroy_vm(vm)


ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8080)
ap.add_argument("--max-vms", type=int, default=3)
ap.add_argument("--boot-wait", type=int, default=12)
A = ap.parse_args()
if os.geteuid() != 0: raise SystemExit("run as root: sudo python3 server.py")
if not (KERNEL and ROOTFS and KEYS): raise SystemExit("fetch Lab 1 assets first")
ensure_bridge()
srv = http.server.ThreadingHTTPServer(("0.0.0.0", A.port), H)
srv.daemon_threads = True
print(f"faas on :{A.port} (bridge {BR}) — POST /run", flush=True)
srv.serve_forever()
