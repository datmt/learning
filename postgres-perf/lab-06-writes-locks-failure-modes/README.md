# Lab 06 — Writes, locks & failure modes

**Question:** it worked on my laptop — why is prod deadlocked / blocked / timing out?

**Answer:** reads scale by adding indexes; writes scale by *shortening
transactions and ordering lock acquisition*. Every failure below is reproduced
live in the demo with real concurrent connections — 2-second timeouts so the
lab stays fast.

```bash
python3 demo.py
```

## 1. Row locks block (and `lock_timeout` saves you)

```sql
-- conn A: BEGIN; UPDATE accounts SET balance=... WHERE id=1;  -- holds row lock, idle
-- conn B: UPDATE accounts ... WHERE id=1;  -- waits... forever by default
```

Default `lock_timeout = 0` = wait forever → thread pools fill with waiters →
cascading outage. Production: `SET lock_timeout = '3s'` (per role/db) so
waiters fail fast with `SQLSTATE 55P03`, then retry with backoff. Demo shows
conn B timing out after 2 s while A still holds the lock.

Diagnose live blockers with:

```sql
SELECT blocked.pid AS waiting, blocking.pid AS holding, blocked.query
FROM pg_stat_activity blocked JOIN pg_stat_activity blocking
  ON blocking.pid = ANY (pg_blocking_pids(blocked.pid));
```

## 2. Deadlocks — inconsistent lock order

A locks row 1 then wants row 2; B locks row 2 then wants row 1 → Postgres
picks a victim (`SQLSTATE 40P01 deadlock_detected`). Your app **must retry**
deadlock victims — they're random. Prevention: always lock rows in a
consistent order (`ORDER BY id` + `SELECT ... FOR UPDATE`), keep transactions
short. Demo fires both orders simultaneously and catches the `40P01`.

## 3. Queue pattern — `FOR UPDATE SKIP LOCKED`

Polling a jobs table with N workers double-processes rows — unless:

```sql
BEGIN;
SELECT id FROM jobs WHERE status='queued' ORDER BY id LIMIT 5
FOR UPDATE SKIP LOCKED;   -- silently skip rows locked by other workers
-- process, UPDATE jobs SET status='done' WHERE id IN (...);
COMMIT;
```

`SKIP LOCKED` (not `NOWAIT`) is the whole trick: contended rows are skipped,
not errored. Demo runs 4 threads × 20 jobs; every job is claimed exactly
once. This is the standard Postgres-backed queue — no external broker needed
at moderate scale.

## 4. Idle-in-transaction — the silent VACUUM blocker

```python
con.execute("BEGIN")     # or Django's ATOMIC_REQUESTS left open
con.execute("SELECT ...")
# ... app does 30 s of HTTP calls, holding the snapshot ...
```

An open transaction pins `xmin`: `VACUUM` can't remove dead rows newer than
it → table bloats (lab 03's bloat, now unbounded) and `pg_stat_activity`
shows `idle in transaction` with an ancient `xact_start`. Production guards:
`idle_in_transaction_session_timeout = '30s'`, transaction-per-request, never
hold a txn across network calls. Demo holds one open and shows the
`pg_stat_activity` row, then rolls back.

## 5. `statement_timeout` — bound the blast radius

One bad deploy (`WHERE` without index, lab 01) + traffic spike = every
connection stuck in Seq Scans. `SET statement_timeout = '5s'` converts
"entire pool wedged for minutes" into "some 500s for 5 seconds". Set it per
role: short for web (`5s`), long for reporting (`10min`). Demo sets 200 ms
and runs `pg_sleep(1)` → `SQLSTATE 57014 query_canceled`.

## Production rules (tape to your monitor)

1. Keep transactions **short**; never hold across I/O.
2. Lock rows in **consistent order** (`ORDER BY pk`); retry `40P01` + `55P03`.
3. Set `lock_timeout`, `statement_timeout`, `idle_in_transaction_session_timeout`.
4. Queues: `FOR UPDATE SKIP LOCKED`, never `SELECT` then `UPDATE` in two trips.
5. Find blockers with `pg_blocking_pids`, slow queries with `pg_stat_statements` (lab 07).

## Exercises

1. Remove `ORDER BY id` from the queue claim — still correct? (Yes, but workers
   collide more; watch throughput drop. Ordering minimizes overlap.)
2. Change the deadlock demo so both sides lock in the same order. Deadlock gone?
3. Set `lock_timeout='100ms'` and re-run §1. Which error? What should your
   app do with it? (Backoff + retry, alert on rate.)
