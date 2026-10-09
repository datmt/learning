# Lab 9 — Production Operations

**Question:** SQLite has no DBA tools and no server. How do I back it up, shrink it, check it, and migrate it?

**Answer:** all built in, via SQL and pragmas. But the obvious approach (`cp app.db backup.db`) is wrong.

```bash
python3 demo.py
```

## 1. Backups

| Method | Safe while app runs? | Notes |
|---|---|---|
| `cp app.db` | **No** | Demo: copy has *no table at all* — 10 MB of committed data was in `app.db-wal` |
| `cp app.db app.db-wal` | **No** | Files can change between the two copies → inconsistent pair |
| Backup API: `conn.backup(dst)` / CLI `.backup file` | Yes | Page-by-page consistent snapshot. Python `pages=N` copies in steps |
| `VACUUM INTO 'file'` (3.27+) | Yes | Consistent + compacted + defragmented copy |
| `sqlite3_rsync` / Litestream (external) | Yes | Incremental / continuous replication |

Then **test the restore**: open the backup and run `PRAGMA integrity_check`.

## 2. File size and VACUUM

- `DELETE` marks pages as free (**freelist**). File size does not change. New inserts reuse them.
- `VACUUM` rebuilds the whole file: 10.27 MB → 1.03 MB in the demo.
  Costs: exclusive lock for the whole run, temp space up to 2x the db size, and time on big dbs.
  In WAL mode the rewritten pages go to `-wal` first; the main file shrinks after a checkpoint.
- `PRAGMA auto_vacuum = INCREMENTAL` (set **before** creating tables, or run `VACUUM` after setting it)
  + `PRAGMA incremental_vacuum(N)` gives space back in small chunks.
  **Quirk:** each step of the statement frees one page. In Python you must `.fetchall()` —
  the demo shows `execute()` alone freed 1 page, with `fetchall()` 1000 pages.
- Don't VACUUM on a schedule "just because". Do it after big deletes if disk space matters.

## 3. Integrity

- `PRAGMA quick_check` — O(n), skips index-vs-table cross checks. Good for frequent runs.
- `PRAGMA integrity_check` — full check. Run on backups and after crashes.
- Demo overwrites 4 KB in the middle of the file → `database disk image is malformed`.

Common real causes of corruption (from sqlite.org "How To Corrupt"): copying a live db, network
filesystems, broken fsync on cheap storage/VMs, `synchronous=OFF`, two copies of SQLite in one
process, deleting a hot `-wal`/`-journal` file. Never delete `-wal`/`-shm` files by hand.

## 4. Statistics: `PRAGMA optimize`

Runs `ANALYZE` only on tables that need it (lab 6 showed why stats matter).
Run it when a long-lived connection closes, or every few hours. `PRAGMA optimize(-1)` shows what it would do.
(3.46+: also good to run right after opening: `PRAGMA optimize=0x10002`.)

## 5. Schema migrations with `user_version`

`PRAGMA user_version` is a free integer stored in the db header. Keep a list of migration
statements; apply those above the current version, each in a transaction **together with**
the version bump (SQLite DDL is transactional — a failed migration rolls back fully).

## Production checklist (everything from labs 1–9)

```python
def connect(path):
    c = sqlite3.connect(path, autocommit=True, timeout=5.0)  # busy_timeout 5 s
    c.execute("PRAGMA journal_mode = WAL")      # readers don't block writer (lab 4)
    c.execute("PRAGMA synchronous = NORMAL")    # fast + no corruption in WAL (lab 5)
    c.execute("PRAGMA foreign_keys = ON")       # off by default! (lab 2)
    c.execute("PRAGMA cache_size = -64000")     # 64 MB cache (optional)
    c.execute("PRAGMA temp_store = MEMORY")     # (optional)
    return c
```

- `STRICT` tables, `INTEGER PRIMARY KEY` (labs 1–2).
- `BEGIN IMMEDIATE` for writes, short transactions, one writer queue (lab 3).
- Batch writes (lab 5). Check `EXPLAIN QUERY PLAN`, run `PRAGMA optimize` (lab 6).
- Backups with backup API / `VACUUM INTO`, verify with `integrity_check` (lab 9).
- Local disk only — never NFS/SMB. Monitor `-wal` size.

## When NOT to use SQLite

- Many app servers on different machines writing to one database → use Postgres/MySQL.
- Sustained high write concurrency where the single writer is the bottleneck.
- You need DB-level users/permissions.

## Exercise

1. Write a `restore_test(backup_path)` that opens a backup read-only (`file:...?mode=ro`, `uri=True`) and runs `integrity_check`.
2. Add a third migration that fails on purpose (`CREATE TABLE t ...`). Confirm `user_version` stays at 2.
