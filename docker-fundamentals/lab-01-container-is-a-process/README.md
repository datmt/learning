# Lab 1 — The Container Is Just a Process

**Question:** when you run `docker run nginx`, where does it *actually* run?

**Answer:** on your host, as an ordinary Linux process. Docker just dresses it up so it
*believes* it is alone on its own machine.

## 0. Beginner primer: what is a process?

A **process** is a running program. The kernel gives every process:

- a number: **PID** (process ID), and a parent's number: **PPID**
- memory, open files, and a spot in the scheduler (so it gets CPU time)

Try without Docker first:

```bash
sleep 10000 &
echo $!              # PID of the background job
ps -o pid,ppid,comm -p $!
kill $!              # clean up
```

`ps aux | grep sleep` lists processes. `PID 1` is special: on a normal Linux system it is
`systemd` (or `init`) — the ancestor of everything.

Key mental picture:

```text
host
├── PID 1  systemd
├── PID 200 sshd
├── PID 500 dockerd
└── PID ???? sleep 10000   <-- just another process
```

Docker will add one more branch like that last line. Nothing magical — yet.

## 1. What you will see

```text
HOST view                          CONTAINER view (docker exec)
─────────────────                  ───────────────────────────
PID 18234 = sleep 10000            PID 1 = sleep 10000   (same process!)
PPID = containerd-shim             PPID = 0 (no parent in its world)
```

Same kernel task, two numbers. Lab 2 explains *how* (PID namespaces). For now just prove it.

## 2. Run it

```bash
./demo.sh      # starts `demo` container, shows host PID vs container PID 1
./inspect.sh   # digs into /proc/<pid>, parent chain, docker top/inspect
./cleanup.sh   # docker rm -f demo
```

## 3. Manual walkthrough (what demo.sh does)

Terminal 1 — start a long-lived container:

```bash
docker run --rm --name demo ubuntu:22.04 sleep 10000
```

Terminal 2 — find it from the host:

```bash
docker top demo
# UID   PID    PPID ... CMD
# root  18234  18212    sleep 10000

docker inspect demo --format '{{.State.Pid}}'   # host PID, e.g. 18234
ps -o pid,ppid,comm -p 18234                    # plain host process!
ls -l /proc/18234/exe                           # points at /usr/bin/sleep
cat /proc/18234/cmdline | tr '\0' ' '           # sleep 10000
```

Hop *inside*:

```bash
docker exec -it demo bash
ps aux        # if ps exists; ubuntu:22.04 needs: apt-get update && apt-get install -y procps
echo $$       # shell PID inside
cat /proc/1/cmdline | tr '\0' ' '   # sleep 10000 — PID 1 here == PID 18234 out there
exit
```

Compare:

| Where | Command | Result |
|-------|---------|--------|
| host | `docker top demo` | host PID (e.g. 18234) |
| host | `ps -p 18234` | same process, `sleep` |
| container | `cat /proc/1/cmdline` | same `sleep 10000` |

## 4. The chain: who started whom?

```bash
ps -o pid,ppid,comm -p $(docker inspect demo --format '{{.State.Pid}}')
ps -ef | grep -E 'dockerd|containerd-shim|runc' | grep -v grep
pstree -p -s $(docker inspect demo --format '{{.State.Pid}}')  # if pstree installed
```

Expect roughly:

```text
systemd(1) ── dockerd ── containerd ── containerd-shim ── sleep 10000
```

So `docker CLI → dockerd → containerd → containerd-shim → runc → kernel`.
`runc` already exited after setup — the leftover `sleep` talks directly to the kernel.
Docker is **not** proxying every syscall.

## 5. Check your understanding

1. `docker top demo` shows PID 18234. Inside, `echo $$` shows 1. Contradiction? (No — namespaces, next lab.)
2. Kill the host PID with `kill 18234`. What happens to `docker ps`? (Container dies — proof it *was* that process.)
3. Run `uname -a` on host and in `docker run --rm ubuntu:22.04 uname -a`. Why identical? (Shared kernel — no guest kernel.)

## 6. Cleanup

```bash
./cleanup.sh
# equivalent: docker rm -f demo
```

Next: **Lab 2** makes the PID illusion explicit and shows you how to create it yourself with `unshare`.
