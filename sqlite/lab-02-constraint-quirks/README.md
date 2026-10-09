# Lab 2 — Constraint and SQL Quirks

**Question:** I wrote valid SQL with a foreign key. Why did SQLite accept garbage data?

**Answer:** SQLite keeps old behaviors forever for backward compatibility. Several of them are
"off" or "loose" by default. You must turn them on yourself.

```bash
python3 demo.py
```

## Quirk list (each one is printed by the demo)

| # | Quirk | Fix |
|---|---|---|
| 1 | `FOREIGN KEY` constraints are **ignored** unless `PRAGMA foreign_keys = ON`. Setting is **per connection**, not saved in the file. | Run the pragma on every new connection. Use `PRAGMA foreign_key_check` to find old orphans. |
| 2 | Every table has a hidden 64-bit `rowid`. Only the exact spelling `INTEGER PRIMARY KEY` makes your column an alias of it. `INT PRIMARY KEY` creates a separate index (`sqlite_autoindex_*`) = extra storage + slower lookups. | Write `INTEGER PRIMARY KEY`. |
| 3 | Without `AUTOINCREMENT`, new id = `max(id)+1`, so deleting the last row lets its id be **reused**. | Use `AUTOINCREMENT` only if reuse is a real problem (e.g. ids leaked to outside systems). It costs an extra write to `sqlite_sequence`. |
| 4 | A non-integer `PRIMARY KEY` **accepts NULL** (old bug kept on purpose). Demo inserts 2 NULL keys. | Add `NOT NULL`, or use `STRICT` / `WITHOUT ROWID` tables which enforce it. |
| 5 | `UNIQUE` allows many NULLs (NULL ≠ NULL). | Add `NOT NULL` if that matters. |
| 6 | `"double quotes"` mean identifier, but if no such column exists SQLite treats it as a **string**. A typo in a column name gives wrong results instead of an error. | Always use `'single quotes'` for strings. Some builds disable this (`SQLITE_DQS=0`). |
| 7 | `LIKE` is case-insensitive only for ASCII. `'ÄBC' LIKE 'äbc'` = 0. `lower('Ä')` = `'Ä'`. `=` is case-sensitive. | Use `COLLATE NOCASE` for ASCII, store a normalized lowercase column (from your app) for Unicode, or load the ICU extension. |
| 8 | Selecting a non-aggregated column with an aggregate is allowed. With `max()`/`min()` the bare column comes from that row (useful!). Otherwise it's an arbitrary row. | Use it deliberately with `max/min` only. |
| 9 | `ALTER TABLE` is limited: no `ADD COLUMN ... UNIQUE`, no non-constant defaults, no changing a column's type. `DROP COLUMN` exists since 3.35. | For big changes do the "12-step" rebuild: create new table → copy → drop old → rename, inside one transaction. |

## Production tips: the "connection init" checklist

Run this on **every** new connection (most settings are not stored in the file):

```sql
PRAGMA foreign_keys = ON;
PRAGMA busy_timeout = 5000;     -- lab 3
PRAGMA journal_mode = WAL;      -- lab 4 (this one IS persistent)
PRAGMA synchronous = NORMAL;    -- lab 5
```

## Exercise

1. Open a second connection to a file DB after turning `foreign_keys` on in the first. What does `PRAGMA foreign_keys` return in the second?
2. Create `CREATE TABLE t (code TEXT PRIMARY KEY) WITHOUT ROWID` and try inserting `NULL`.
