# Lab 8 — Hard Limits

**Question:** how big can a SQLite database get? How many columns, parameters, rows?

**Answer:** much bigger than most apps need — but some limits are small enough to hit in
normal code (bound parameters, compound SELECT, ATTACH). The demo pushes each limit until it errors.

```bash
python3 demo.py
```

## 1. The limits (this build: SQLite 3.51.3 bundled with Python 3.14)

| Limit | Value here | Upstream default | Error when exceeded |
|---|---|---|---|
| String/BLOB size (`LENGTH`) | 1,000,000,000 bytes | same (hard max 2^31-1) | `string or blob too big` |
| SQL statement length | 1,000,000,000 bytes | same | Python: `DataError: query string is too large` (C API: `statement too long`) |
| Columns per table/index/SELECT | 2,000 | same (hard max 32,767) | `too many columns on wider` |
| Bound parameters `?` | **250,000** | **32,766** (999 before 3.32!) | `too many SQL variables` |
| Expression depth | 1,000 | same | `Expression tree is too large` |
| Terms in `UNION ALL` chain | 500 | same | `too many terms in compound SELECT` |
| Attached databases | 10 | same (max 125) | `too many attached databases` |
| Integer | 64-bit signed | | overflow → REAL (lab 1) |
| Page size | 512 – 65,536 bytes (default 4,096) | | |
| Max pages | 4,294,967,294 | | `database or disk is full` |
| **DB size** | 16 TiB at 4 KiB pages, **256 TiB** at 64 KiB pages | | |
| rowid | max 2^63-1 | | random free rowid; with AUTOINCREMENT → `database or disk is full` |

`LENGTH`, `VARIABLE_NUMBER` etc. vary by build. Your phone's or distro's SQLite may differ.
Always check: `conn.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)` (Python 3.11+).
`conn.setlimit()` can **lower** limits at runtime (useful for untrusted SQL), never raise past the compile-time max.

## 2. Limits you will actually hit

1. **`WHERE id IN (?, ?, ...)` with a huge list** → `too many SQL variables` on older/other builds.
   Fix: pass one JSON array: `WHERE id IN (SELECT value FROM json_each(?))`, or fill a temp table.
2. **Multi-row insert built with `UNION ALL SELECT`** → 500 terms max. Use `VALUES (...), (...)` (no limit besides SQL length) or `executemany`.
3. **Disk full** → `database or disk is full`. `PRAGMA max_page_count` lets you cap the file size on purpose (demo part 9).
4. **Big BLOBs** → a 1 GB value must fit in memory in one piece. Store large files on disk / object storage and keep the path in SQLite. Rule of thumb from SQLite docs: BLOBs under ~100 KB are faster in SQLite than as separate files.

## 3. Practical (not hard) limits

- **One writer at a time** (labs 3–4). Typical ceiling: thousands to tens of thousands of write *transactions*/s, millions of rows/s if batched.
- **Single machine.** No built-in replication/clustering. (Tools exist: Litestream, LiteFS, rqlite — out of scope.)
- **Size**: multi-TB SQLite databases work, but `VACUUM`, backup and schema changes become slow. Sweet spot: up to ~hundreds of GB.

## Exercise

1. Use `db.setlimit(sqlite3.SQLITE_LIMIT_SQL_LENGTH, 1000)` and write a query longer than 1000 bytes. What error?
2. Create a db with `PRAGMA page_size = 65536` *before* creating any table. Verify with `PRAGMA page_size`.
