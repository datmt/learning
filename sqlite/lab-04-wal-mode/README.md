# Lab 4 — WAL Mode

**Question:** lab 3 showed readers blocking writers. How do real apps run SQLite under load?

**Answer:** `PRAGMA journal_mode = WAL`. Readers and the writer stop blocking each other.
Still **one writer at a time**.

```bash
python3 demo.py
```

## 1. How it works

```text
rollback journal (DELETE):  copy OLD pages to db-journal, write NEW pages into db file
WAL:                        append NEW pages to db-wal; db file untouched until "checkpoint"

            ┌──────────────┐  reads: db file + newest version of page in WAL (up to its snapshot)
 readers ──▶│  db   │ -wal │
 writer  ──▶│       │ ▲    │  writes: append only to -wal
            └───────┴─┼────┘
          checkpoint ─┘ copies WAL pages back into db file
```

- `-wal`: the log. `-shm`: shared-memory index of the WAL (so readers find pages fast).
- A reader picks a **snapshot** (point in the WAL) when its transaction starts and keeps seeing it.

## 2. What the demo shows (sample run, Linux, NVMe)

```text
== 1. Long reader vs writer
   DELETE: writer FAILED: database is locked        reader saw (1,) -> (1,) inside txn, (1,) after
   WAL   : writer committed                         reader saw (1,) -> (1,) inside txn, (2,) after

== 2. Throughput: 1 writer + 4 busy readers for 2 s
   DELETE: {'writes': 2121, 'reads': 8, 'busy': 0}
   WAL   : {'writes': 4244, 'reads': 35609, 'busy': 0}

== 3. Checkpoints and WAL growth
   after 5 MB of inserts (autocheckpoint ran): db=3,821,568  -wal=4,157,112
   10 MB more while a reader is stuck open:      db=5,025,792  -wal=12,228,192
   checkpoint(PASSIVE) busy,log,done: (0, 2968, 318)
   checkpoint(TRUNCATE) after reader leaves:  (0, 0, 0)
```

Readers: 8 → 35,609 (**~4000x**). Writer: 2x (WAL appends sequentially, fewer fsyncs).
Your numbers will differ (Python GIL + disk), but the ratio is the lesson.

## 3. Checkpoints

- Auto-checkpoint runs when WAL reaches `wal_autocheckpoint` pages (default 1000 ≈ 4 MB).
- A checkpoint can only copy frames **older than the oldest active reader's snapshot**.
  Part 3: an open read transaction → only 318 of 2968 frames copied → WAL grows to 12 MB.
  With a reader that never closes, **the WAL grows forever** ("checkpoint starvation").
- The WAL file is **reused, not shrunk** after a normal checkpoint (stays ~4 MB).
  `PRAGMA journal_size_limit = N` or `wal_checkpoint(TRUNCATE)` shrinks it.

Checkpoint modes: `PASSIVE` (don't wait, default for auto), `FULL` (wait for writers),
`RESTART` (also wait for readers so the WAL restarts from the top), `TRUNCATE` (RESTART + truncate file to 0).

## 4. WAL limits and gotchas

- **Persistent**: setting stored in the db file; every later connection uses WAL.
- **Same machine only**: `-shm` is shared memory. WAL does **not** work on network filesystems (NFS/SMB).
  (Rollback mode on NFS is also unsafe due to broken file locks. Don't put SQLite on NFS.)
- Very large write transactions (GBs) make a huge WAL. Split them.
- Read-only media: a WAL db needs `-shm` write access (or `?immutable=1`).
- Backups: copying only `app.db` while a `-wal` exists **loses committed data**. Use lab 9's backup methods.

## Production tips

```sql
PRAGMA journal_mode = WAL;
PRAGMA synchronous = NORMAL;      -- safe in WAL (lab 5)
PRAGMA journal_size_limit = 67108864;   -- cap leftover WAL at 64 MB
```

- Keep read transactions short; close cursors (`fetchall()`, or `with` blocks) so snapshots release.
- Watch the `-wal` file size in monitoring. Growth = stuck reader.
- Optional: run `PRAGMA wal_checkpoint(TRUNCATE)` from a background job during quiet times.

## Exercise

1. In part 1, replace `BEGIN` with nothing for the reader. Does the reader see `2` immediately? Why?
2. Set `PRAGMA wal_autocheckpoint = 0` and insert 20 MB. Watch the `-wal` file.
