# Lab 3 — Transactions, Locks, `database is locked`

**Question:** why does my app randomly throw `sqlite3.OperationalError: database is locked`?

**Answer:** SQLite allows **one writer at a time for the whole database file**. Other writers
must wait or fail. Default wait time in the C library is **0** (fail instantly). And one
transaction pattern fails instantly *even with* a wait time configured.

```bash
python3 demo.py
```

## 1. Lock states (rollback-journal mode, the default)

```text
UNLOCKED ──read──▶ SHARED ──first write──▶ RESERVED ──commit──▶ PENDING ──▶ EXCLUSIVE
 (many readers can hold SHARED)   (only 1 RESERVED)       (wait for readers to leave) (write file)
```

- Many connections can read at once (`SHARED`).
- One connection can prepare writes (`RESERVED`). Others can still read.
- To **commit**, the writer needs `EXCLUSIVE` → all readers must finish first.

Demo part 2 shows: B can read while A has uncommitted changes, B cannot write,
and A's `COMMIT` fails while B holds an open read transaction.
(Lab 4's WAL mode removes the reader/writer conflict.)

## 2. Transaction types

| Statement | Lock taken at BEGIN | Use for |
|---|---|---|
| `BEGIN` / `BEGIN DEFERRED` (default) | none; SHARED on first read, RESERVED on first write | read-only transactions |
| `BEGIN IMMEDIATE` | RESERVED right away | **any transaction that will write** |
| `BEGIN EXCLUSIVE` | EXCLUSIVE (in WAL same as IMMEDIATE) | rare |

Without `BEGIN`, each statement is its own transaction ("autocommit") — 1 disk sync per statement (see lab 5).

## 3. `busy_timeout`

`PRAGMA busy_timeout = 5000` (or Python `sqlite3.connect(..., timeout=5.0)`) makes SQLite
retry for up to 5 s before raising `database is locked`. Demo part 3: B waits 0.5 s and succeeds.

## 4. The trap: read-then-write in a DEFERRED transaction

```text
A: BEGIN; SELECT ...;  UPDATE ...   (SHARED -> RESERVED)
B: BEGIN; SELECT ...;  UPDATE ...   (SHARED -> wants RESERVED)  => instant SQLITE_BUSY
```

B holds a read snapshot. If B waited for A, A's commit would wait for B's SHARED lock to go away
→ deadlock. So SQLite fails B **immediately, ignoring busy_timeout**. The demo prints `failed after 0.00s`.

In WAL mode the same thing happens (`SQLITE_BUSY_SNAPSHOT`): B's snapshot is stale after A commits.

**Fix:** start write transactions with `BEGIN IMMEDIATE`. The wait happens at `BEGIN`,
where busy_timeout works (demo part 5).

## 5. Errors don't auto-rollback

After a constraint error inside `BEGIN ... COMMIT`, the transaction is **still open**
(`in_transaction: True`). Your code must `ROLLBACK`, or the next `COMMIT` saves the partial work.

## 6. SAVEPOINT

`SAVEPOINT name` / `ROLLBACK TO name` / `RELEASE name` give you nested, partial undo.

## Python notes

- Python ≥ 3.12: `sqlite3.connect(..., autocommit=True)` makes Python behave like plain SQLite.
  The old default (`isolation_level=""`) silently issues `BEGIN` (DEFERRED!) before DML — easy to hit the trap above.
- A connection is single-thread by default. `check_same_thread=False` is used in the demo only because a timer thread commits.
  In real apps use **one connection per thread**.

## Production tips

1. Set `busy_timeout` (5000 ms is common) on every connection.
2. Use `BEGIN IMMEDIATE` for every transaction that writes.
3. Keep write transactions **short**. No network calls inside them.
4. Many writer threads? Funnel writes through a single writer connection/queue. One writer is the SQLite model.
5. Wrap transactions in try/except and always `ROLLBACK` on error.

## Exercise

Remove `check_same_thread=False` and run. What error does Python raise, and why does it protect you?
