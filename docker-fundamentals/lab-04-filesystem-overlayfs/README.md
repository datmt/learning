# Lab 4 — Images & OverlayFS: Where Does `/bin/bash` Come From?

**Question:** `docker run ubuntu` gives you `/bin/bash`, `/usr/lib/...`, `apt` — without booting Ubuntu.
Where do those files live?

**Answer:** in **image layers** stacked by **OverlayFS**: read-only lower layers (the image) +
one writable upper layer (your container's changes). `apt install` or `echo hi > /file` only touches
the upper layer; deleting the container discards it.

## 0. Beginner primer

- **Image**: a frozen template — a stack of tarball-like layers (each `RUN`/`COPY` in a Dockerfile adds one).
- **Layer**: a directory of files + a record of deletions ("whiteouts"). `docker history ubuntu` shows them.
- **OverlayFS**: a Linux *union filesystem* that merges directories:
  - `lowerdir` = image layers (read-only, can be many, colon-separated)
  - `upperdir` = container's scratch space (read-write)
  - `merged` = what the container sees as `/`
  - `workdir` = internal bookkeeping dir OverlayFS needs
- **Copy-up**: first write to a lower-layer file copies it to upperdir, then modifies the copy.
  Lower layers never change → images stay shareable across containers.
- **Volume** (`-v`): bypasses overlay entirely (plain host bind-mount) — for databases etc.

```text
container sees (merged = /)
        │
  ┌─────┴──────┐
  │            │
upperdir    lowerdir(s)        <-- docker inspect → GraphDriver.Data
(container   (image layers,
 writable     read-only, shared)
 layer)
```

## 1. Run it

```bash
./demo.sh      # history + GraphDriver dirs + write test proving upperdir isolation
./inspect.sh   # browse lowerdir/upperdir/merged on the HOST (sudo) + diff/size
./layers.sh    # build a tiny 3-layer image to see layers appear
./cleanup.sh
```

## 2. Manual walkthrough

```bash
docker pull ubuntu:22.04
docker history ubuntu:22.04 | head          # each line = one layer
docker info | grep -i storage               # overlay2

docker run -d --rm --name fsLayer ubuntu:22.04 sleep 10000
docker inspect fsLayer --format '{{json .GraphDriver.Data}}' | python3 -m json.tool
# { LowerDir: "/var/lib/docker/overlay2/<hash>/diff:...", UpperDir: ".../diff",
#   MergedDir: ".../merged", WorkDir: ".../work" }
```

Write test — changes land in upperdir only:

```bash
docker exec fsLayer bash -c 'echo hello > /hello.txt && echo "intruded" >> /etc/hosts-suffix-test 2>/dev/null; echo hello > /tmp/t.txt; ls /'
sudo ls "$(docker inspect fsLayer --format '{{.GraphDriver.Data.UpperDir}}')"   # hello.txt, tmp/t.txt here
sudo ls "$(docker inspect fsLayer --format '{{.GraphDriver.Data.LowerDir}}' | cut -d: -f1)/.." | head  # image hashes, untouched
docker diff fsLayer        # A = added, C = changed (copy-up), D = deleted
```

Two containers share lowers, diverge uppers:

```bash
docker run -d --rm --name fsA ubuntu:22.04 sleep 10000
docker run -d --rm --name fsB ubuntu:22.04 sleep 10000
docker exec fsA touch /only-in-A
docker exec fsB ls /only-in-A 2>&1 || echo "not in B ✔ (separate upperdirs)"
docker inspect fsA --format '{{.GraphDriver.Data.UpperDir}}'
docker inspect fsB --format '{{.GraphDriver.Data.UpperDir}}'   # different dirs
docker rm -f fsA fsB
```

## 3. Dockerfile → layers (see `layers.sh`)

```dockerfile
FROM ubuntu:22.04
RUN echo one > /one.txt
RUN echo two > /two.txt
```

Each `RUN` = one layer in `docker history`. Reordering invalidates build cache below the change —
that's why `COPY requirements.txt` + `RUN pip install` goes *before* `COPY . .` in real Dockerfiles.

## 4. Check your understanding

1. You `apt-get install curl` in a container, then `docker rm` it and start fresh. Where's curl? (Gone — it was in the discarded upperdir. Bake it into the image to keep it.)
2. 10 containers from `ubuntu` use ~10× disk? (No — lowerdirs shared; only upperdir diffs cost space. Check `docker system df`.)
3. Why are DBs put on `-v` volumes, not in upperdir? (Performance + persistence + survives `docker rm`.)

Next: **Lab 5** — networking: how an isolated `eth0` reaches the Internet via `veth` + `docker0` + NAT.
