# Lab 2 — PID Namespaces: Why Am I PID 1 In Here?

**Question:** why does `echo $$` print `1` inside the container but `docker top` shows
e.g. PID 18234 on the host?

**Answer:** a **PID namespace** gives a process group its own private numbering starting at 1.
The kernel keeps the mapping. Same task, two numbers.

## 0. Beginner primer

- **PID**: kernel's ID for a process. Unique *within a PID namespace*.
- **PID 1**: the first process in a namespace — traditionally `init`/`systemd`. Orphaned processes get reparented to it.
- **Namespace**: kernel feature that says "this group of processes sees its own version of X"
  (PIDs, mounts, network, hostname, ...). Created with `clone()`/`unshare()` syscalls.
- **`/proc`**: a fake filesystem the kernel generates — `ps`, `top`, `ls /proc` all read it.
  A container gets its own `/proc` showing only its namespace.

The 7 namespace types that matter for Docker:

| Namespace | Isolates | Docker effect |
|-----------|----------|---------------|
| `pid` | process IDs | container sees itself as PID 1 |
| `mnt` | mounts | own filesystem view (Lab 3) |
| `net` | network | own eth0/IP (Lab 5) |
| `uts` | hostname | own hostname |
| `ipc` | shared memory/semaphores | can't snoop host IPC |
| `user` | UIDs/GIDs | root remapping (rootless Docker) |
| `cgroup` | cgroup view | own resource view |

Picture:

```text
              Linux kernel (one task: sleep)
                         │
              host PID 18234
                         │
            ┌────────────┴────────────┐
            │                         │
     host PID namespace        container PID namespace
     sees 18234                   sees 1
     sees ALL processes           sees ONLY its own
```

## 1. Run it

```bash
./demo.sh      # container + host/container PID comparison + ns inodes
./no-docker.sh # SAME illusion with pure Linux: sudo unshare --pid --fork --mount-proc bash
./inspect.sh   # nsenter into container namespaces from host
./cleanup.sh
```

## 2. Manual walkthrough

```bash
docker run -d --rm --name piddemo ubuntu:22.04 sleep 10000
HOST_PID=$(docker inspect piddemo --format '{{.State.Pid}}')
echo "host PID: $HOST_PID"

# Host sees the real number:
ps -o pid,ppid,comm -p "$HOST_PID"

# Container sees 1:
docker exec piddemo cat /proc/1/cmdline | tr '\0' ' '; echo
docker exec piddemo bash -c 'echo $$'   # shell inside, e.g. 20-something

# Namespace inodes differ (different worlds) — no sudo needed via docker exec:
readlink /proc/self/ns/pid                    # your shell
docker exec piddemo readlink /proc/1/ns/pid    # container init — different inode ✔

# (Optional, needs sudo password) step INTO the container's PID world from the host:
sudo nsenter -t "$HOST_PID" -p ps aux   # -p = PID namespace; now host shows container PIDs!
```

Kill test (proves identity):

```bash
kill "$HOST_PID"          # kills "PID 1" inside too
docker ps --filter name=piddemo   # gone
```

## 3. No-Docker version (the money exercise)

This uses only Linux — no Docker daemon involved:

```bash
sudo unshare --pid --fork --mount-proc bash
echo $$          # 1 !
ps aux           # only your shell + ps
exit             # back to host; numbering normal again
```

Flags: `--pid` = new PID ns, `--fork` = required so the child becomes PID 1,
`--mount-proc` = fresh `/proc` so `ps` reads the new world.

You just did ~10% of what `runc` does.

## 4. Check your understanding

1. Why must `unshare --pid` use `--fork`? (The calling shell keeps its old PID; only a *child* can be PID 1 in the new ns.)
2. `docker exec piddemo ps aux` shows few processes; host `ps aux` shows hundreds. Which one is "true"? (Both — different namespace views of one kernel process table.)
3. Why can't a container `kill 1` escape to kill host init? (PID 1 in container maps to e.g. 18234 on host; `kill 1` hits container init only.)

Next: **Lab 3** — same trick, but for filesystems (mount namespaces + `mount`/`df`).
