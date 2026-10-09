# SQLite — Capabilities, Quirks, and Production Tuning

Hands-on labs. Each lab = `README.md` (theory) + `demo.py` (runs every scenario and prints proof).

**Requirements:** Python ≥ 3.12 (uses `sqlite3.connect(..., autocommit=True)`). Standard library only.
Tested with Python 3.14 / SQLite 3.51.3 on Linux (ext4, NVMe).

```bash
python3 -c "import sqlite3; print(sqlite3.sqlite_version)"   # your SQLite version
cd lab-01-dynamic-typing && python3 demo.py
```

> SQLite is a **library**, not a server. Your app opens a file and reads/writes it directly.
> No network hop, no separate process. That's why it's fast — and why concurrency works differently.

## Lab map

| Lab | Question it answers |
|---|---|
| [01 dynamic typing](lab-01-dynamic-typing/) | Why can I store `'hello'` in an `INTEGER` column? What do `STRICT` tables fix? |
| [02 constraint quirks](lab-02-constraint-quirks/) | Why are my foreign keys ignored? NULL primary keys? Double-quote strings? |
| [03 transactions & locking](lab-03-transactions-locking/) | Where does `database is locked` come from? Why does `busy_timeout` sometimes not help? |
| [04 WAL mode](lab-04-wal-mode/) | How do readers and a writer run together? Why does my `-wal` file keep growing? |
| [05 write performance](lab-05-write-performance/) | 1,000 rows/s or 2,000,000 rows/s? `synchronous`, batching, key choice |
| [06 indexes & query planner](lab-06-indexes-query-planner/) | Is my index used? Covering, composite, partial, expression indexes, `ANALYZE` |
| [07 modern features](lab-07-modern-features/) | JSON, FTS5 full-text search, window functions, CTEs, UPSERT, RETURNING |
| [08 limits](lab-08-limits/) | Max columns, parameters, BLOB size, DB size — push each until it breaks |
| [09 production ops](lab-09-production-ops/) | Backups that work, VACUUM, integrity checks, migrations, final checklist |

## Key numbers measured in these labs

| Scenario | Result |
|---|---|
| Insert, 1 txn per row, default settings (rollback + `synchronous=FULL`) | ~1,000 rows/s |
| Insert, 1 txn per row, `WAL` + `synchronous=NORMAL` | ~120,000 rows/s |
| Insert, 1 txn for all rows, `executemany` | ~2,000,000+ rows/s |
| 4 readers during writes: rollback journal vs WAL | 8 vs 35,609 reads in 2 s |
| Lookup with vs without index (500k rows) | 0.01 ms vs 10 ms |

## Top 10 production rules

1. `PRAGMA journal_mode = WAL` (persistent).
2. `PRAGMA synchronous = NORMAL` in WAL (keep `FULL` if every commit must survive power loss).
3. `PRAGMA busy_timeout = 5000` on every connection.
4. `PRAGMA foreign_keys = ON` on every connection.
5. `BEGIN IMMEDIATE` for transactions that write; keep them short.
6. Batch many writes per transaction.
7. `STRICT` tables + `INTEGER PRIMARY KEY`.
8. Check `EXPLAIN QUERY PLAN`; run `PRAGMA optimize`.
9. Back up with the backup API or `VACUUM INTO`, never `cp` a live db.
10. Local disk only. One machine. If you need many writer machines, use a server database.
