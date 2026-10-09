# Lab 05 — Joins & aggregations at scale

**Question:** my join / `GROUP BY` is fine on 10k rows and dies on 10M — what changed?

**Answer:** the algorithm. Small inputs use Nested Loops and in-memory sorts;
large inputs need Hash Joins and disk spills — and `work_mem` decides the
boundary. Plus two classic correctness traps (`IN` + NULL, CTE surprises) that
only bite at scale. This lab forces each regime so you recognize the plan
shape in production.

```bash
python3 demo.py              # default 300k orders / 20k customers
SCALE=1000000 python3 demo.py
```

## 1. Join algorithms — size picks the winner

| Algorithm | Shape | Good when |
|---|---|---|
| Nested Loop | for each outer row, index-probe inner | outer side tiny (≤ hundreds), inner indexed |
| Hash Join | hash small side, scan big side once | large joins, equi-join, enough `work_mem` |
| Merge Join | both sides pre-sorted, one merge pass | inputs already indexed/sorted, `ORDER BY` compatible |

The planner estimates sizes and picks. The failure mode: a Nested Loop with
100k outer rows × index probe = 100k random reads (fine at 100 rows, outage
at 100k). Demo runs the same join with 10 vs 100k outer rows so you see the
switch — and what happens when a bad estimate picks wrong (lab 01's stale
stats + this lab = full story).

## 2. `work_mem` + disk spills — the sort cliff

`work_mem` (default 4 MB) is **per sort/hash operation**: exceed it and
Postgres spills to disk (`Sort Method: external merge  Disk: 1234kB`,
`temp written` in BUFFERS, `log_temp_files` in logs). Runtime falls off a
cliff — several × slower, and it thrashes shared disks. Same mechanism hits
`GROUP BY` hashes and hash joins, not just `ORDER BY` sorts.

Demo: `SET work_mem='64kB'` forces the spill on an `ORDER BY` of 300k rows,
then resets to 4 MB to show it vanish. (Side lesson: the first attempt grouped
by the *indexed* column and Postgres dodged the sort entirely via the index —
indexes can save you from sorts, which is also why lab 02's composite order
matters.) Production moves: raise `work_mem` for the
reporting role only (`SET` per session, not globally — 100 connections ×
1 GB `work_mem` = OOM), pre-aggregate (materialized view, lab 07), or
partition the GROUP BY key.

## 3. N+1 — the app-side join failure

```python
for c in customers:                      # 1 + N queries
    db.execute("SELECT ... WHERE customer_id=%s", c.id)
```

Demo: 200 PK lookups one-by-one (~200 round trips) vs one `JOIN`/`IN`
(~1 trip) — typically 20–50×. Fix with a single join, `WHERE id = ANY(%s)`,
or batching (DataLoader pattern). `pg_stat_statements` (lab 07) exposes N+1
as "same query, huge calls, tiny rows each".

## 4. `EXISTS` vs `IN` (NULL trap + perf)

- `WHERE x IN (SELECT ...)` with a **NULL** in the subquery → `NOT IN`
  returns *nothing* (NULL poisons three-valued logic). `NOT EXISTS` is safe.
- `EXISTS` stops at the first match (semi-join); `IN` may materialize the
  whole set. Planner often rewrites both to the same Semi Join — verify with
  EXPLAIN, prefer `EXISTS` for "has at least one" checks.

## 5. CTEs: optimization fence (history lesson)

Pre-PG12, `WITH x AS (...)` always materialized (ran fully, then joined) —
a fence. Since PG12, CTEs inline like subqueries unless marked
`MATERIALIZED`. Demo shows both forms: use `MATERIALIZED` deliberately when
the CTE is referenced 3× and expensive (compute once), `NOT MATERIALIZED`
(or plain subquery) when filtering should push down.

## Exercises

1. Re-run the spill demo with `work_mem='1MB'` vs `'64kB'`. At what size does
   *your* GROUP BY spill? (`EXPLAIN` shows `Sort Method`.)
2. Force the wrong join: `SET enable_hashjoin=off` on the big join — time it.
   Then reset. When would you *ever* ship that? (Almost never.)
3. Add a NULL `customer_id` row and compare `NOT IN` vs `NOT EXISTS` results.
