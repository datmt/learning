# Lab 5 — Write Performance

**Question:** people say SQLite is slow at inserts: ~100 rows/second. Others say millions/second. Who is right?

**Answer:** both. Speed is decided by **how many transactions** (= disk syncs) you do, not how many rows.

```bash
python3 demo.py
```

> The demo writes its temp DBs **inside this folder**, not `/tmp`. On many Linux systems `/tmp`
> is `tmpfs` (RAM), where `fsync` costs nothing and the results lie.

## 1. Sample run (Linux, ext4, NVMe SSD)

```text
== 1. 2,000 inserts, one transaction per row vs one transaction total
   DELETE  sync=FULL   row-by-row             2,000 rows    1.938s         1,032 rows/s
   WAL     sync=FULL   row-by-row             2,000 rows    0.556s         3,600 rows/s
   WAL     sync=NORMAL row-by-row             2,000 rows    0.017s       118,760 rows/s
   WAL     sync=OFF    row-by-row             2,000 rows    0.011s       175,527 rows/s
   DELETE  sync=FULL   one txn                2,000 rows    0.002s       928,919 rows/s
   WAL     sync=NORMAL one txn                2,000 rows    0.001s     1,907,025 rows/s

== 2. Bulk load 1,000,000 rows in one transaction
   WAL sync=NORMAL executemany            1,000,000 rows    0.414s     2,417,723 rows/s

== 3. Index before vs after bulk load
   index created before load: 0.804s
   index created after  load: 0.568s

== 4. Random UUID-like keys vs sequential keys (B-tree page splits)
   sequential keys: 0.154s
   random     keys: 0.478s
```

On a spinning disk the row-by-row FULL number drops to ~50–100 rows/s.

## 2. Why: fsync

A commit must survive power loss, so SQLite calls `fsync()` and waits for the disk to confirm.
That costs ~0.1–10 ms. **Rows per commit** decides throughput:

```text
1 row per txn    -> 2,000 fsync calls
2,000 rows / txn -> 1-2 fsync calls          ~1000x faster
```

## 3. `PRAGMA synchronous`

| Value | Rollback (DELETE) mode | WAL mode |
|---|---|---|
| `FULL` (default) | durable, safe | durable, safe; fsync on every commit |
| `NORMAL` | small corruption risk on power loss | **safe from corruption**; last few commits may roll back on *power loss* (not app crash). fsync only at checkpoint |
| `OFF` | corruption risk on OS crash/power loss | same; never fsync |

`WAL + NORMAL` is the standard production choice: ~30x faster than `WAL + FULL` in the demo,
DB never corrupts, you might lose the last ~second of commits if the **machine** loses power.
Need every commit durable (payments)? Keep `FULL`.

## 4. Other tips shown

- **`executemany`** + one transaction: ~2.4M rows/s from Python.
- **Create indexes after bulk loading**: one sorted build beats 1M incremental inserts.
- **Sequential keys beat random keys** (UUIDv4): random inserts touch random B-tree pages,
  causing more page splits and cache misses. 3x slower here, much worse when the table is bigger
  than the cache. Use `INTEGER PRIMARY KEY`, or time-ordered ids (UUIDv7/ULID) if you need UUIDs.

## 5. More knobs (measure before using)

```sql
PRAGMA cache_size = -64000;      -- negative = KiB -> 64 MB page cache per connection (default ~2 MB)
PRAGMA temp_store = MEMORY;      -- temp tables / sort spill in RAM
PRAGMA mmap_size = 268435456;    -- 256 MB memory-mapped reads
```

Only for throw-away bulk imports you can rebuild: `journal_mode=OFF`, `synchronous=OFF`
(a crash mid-import = corrupt file).

## Production tips

1. Batch writes: many rows per `BEGIN IMMEDIATE ... COMMIT`.
2. `journal_mode=WAL`, `synchronous=NORMAL` unless every commit must survive power loss.
3. Reuse connections (prepared statements are cached per connection; Python `cached_statements=128`).
4. Bulk load: drop/skip indexes, load in one txn, then `CREATE INDEX`, then `ANALYZE`.

## Exercise

Change `N` to 10,000 and add a run with 100 rows per transaction. Plot rows/s against rows-per-txn.
Where does the curve flatten?
