# Lab 3 — Mount Namespaces: Why the Container Sees a Different Filesystem

**Question:** why do `mount`, `df -h`, and `/etc/os-release` show completely different things
inside vs outside the container?

**Answer:** a **mount namespace** gives the container its own mount table. Docker mounts an
Ubuntu root filesystem as `/` for that namespace, so `cat /etc/os-release` reads Ubuntu files —
while the kernel underneath is still the host's.

## 0. Beginner primer

- **Mount**: attaching a filesystem at a path (`mount /dev/sda1 /`, `mount -t proc proc /proc`).
- **Mount table**: the list of what's mounted where — see with `mount`, `df -h`, `cat /proc/mounts`.
- **Root filesystem (`/`)**: what a process sees as `/`. Your host might be Arch; the container's `/` is an Ubuntu image tree.
- **`chroot` (old) vs mount namespace (new)**: `chroot` just changes `/` for one process and is escapable; a mount namespace gives an isolated mount table the host can't accidentally see.
- Docker combines: **mount ns** (private table) + **overlayfs** (layered image, Lab 4) + **pivot_root** (swap in container root).

```text
HOST mount table                     CONTAINER mount table (own namespace)
────────────────                     ────────────────────────────────────
/          = Arch rootfs             /          = Ubuntu image (overlay)
/proc      = host procs              /proc      = only container procs
/sys       = host sysfs              /etc/hostname, /etc/hosts = Docker-injected
/home/...  = your disks              (host disks invisible unless -v mounted)
```

## 1. Run it

```bash
./demo.sh      # compares host vs container: mount, df, os-release, /proc/mounts
./no-docker.sh # pure-Linux mount isolation: unshare --mount, bind-mount a tmp dir
./inspect.sh   # docker inspect mounts + nsenter into container mount ns
./cleanup.sh
```

## 2. Manual walkthrough

```bash
docker run -d --rm --name mntdemo ubuntu:22.04 sleep 10000

# Filesystem identity:
cat /etc/os-release | head -3              # host (Arch)
docker exec mntdemo cat /etc/os-release | head -3   # Ubuntu 22.04 — image tree, not a boot!

# Mount tables diverge:
mount | head -20
docker exec mntdemo mount | head -20
df -h
docker exec mntdemo df -h

# Docker injects a few files into every container:
docker exec mntdemo cat /etc/hostname
docker exec mntdemo cat /etc/hosts
docker exec mntdemo cat /etc/resolv.conf
docker inspect mntdemo --format '{{json .Mounts}}' | python3 -m json.tool
```

Prove host files are hidden (unless explicitly mounted):

```bash
echo secret > /tmp/host-only.txt
docker exec mntdemo cat /tmp/host-only.txt || echo "NOT VISIBLE — mount isolation works"
docker run --rm -v /tmp:/tmp ubuntu:22.04 cat /tmp/host-only.txt  # opt-in via -v
rm /tmp/host-only.txt
```

## 3. No-Docker version

```bash
mkdir -p /tmp/mntdemo && echo hello > /tmp/mntdemo/file.txt
sudo unshare --mount --fork bash
mount --bind /tmp/mntdemo /mnt
ls /mnt            # file.txt visible only in this namespace
# (open another terminal: ls /mnt is empty there)
exit
```

You made a private mount — the seed of what Docker does with whole root filesystems.

## 4. Check your understanding

1. `cat /etc/os-release` differs but `uname -a` is identical — why? (Filesystem comes from image; kernel is shared. Files ≠ kernel.)
2. Where do `/etc/hostname`/`/etc/hosts` inside come from? (Docker bind-mounts generated files per container — check `docker inspect` → `HostnamePath`, `HostsPath`.)
3. What does `-v /host:/ctr` do in namespace terms? (Adds a bind-mount entry to the container's mount table only.)

Next: **Lab 4** opens up that Ubuntu filesystem — image layers and OverlayFS.
