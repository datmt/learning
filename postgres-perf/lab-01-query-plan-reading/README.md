# Lab 01 — Reading query plans (`EXPLAIN`)

**Question:** the query is slow — what is Postgres *actually doing*?

**Answer:** `EXPLAIN` prints the planner's chosen tree of nodes
(Scan → Join → Sort → Limit). `EXPLAIN ANALYZE` runs it and annotates each
node with real time + real rows. Your job: compare **estimated rows vs actual
rows**; a big gap means bad stats → wrong plan. 90% of "slow query" debugging
is this comparison.

```bash
python3 demo.py              # default 200k rows, ~2 s seed
SCALE=1000000 python3 demo.py
```

## 1. The vocabulary (only 6 nodes cover most queries)

| Node you see | Meaning | When it's good / bad |
|---|---|---|
| `Seq Scan` | read every row, filter | fine for tiny tables or "return 95% of rows"; disaster for "1 row out of 1M" |
| `Index Scan` | walk B-tree, fetch rows in order | good for selective `WHERE` + `ORDER BY ... LIMIT` |
| `Index Only Scan` | answer from the index, never touch the table | best case (needs covering index + `VACUUM`, lab 02) |
| `Bitmap Heap Scan` | collect matching row pointers, then read table in bulk | good for medium selectivity (1–10%); the planner's compromise |
| `Nested Loop` | for each outer row, probe inner | good for "few outer rows × indexed inner"; terrible for "100k × 100k" |
| `Hash Join` / `Merge Join` | build hash / use sorted order, one pass each side | good for large joins; hash needs `work_mem` |

`cost=0.00..1234.56` is an **abstract unit**, not milliseconds — only use it
to compare plans for the *same* query. `actual time=0.5..120.3` **is**
milliseconds (startup..total).

## 2. How to read any plan in 30 seconds

1. Read **inside-out, bottom-up**: the indented leaves run first.
2. For each node check `rows (est)` vs `rows (actual)`. Off by > 10×? Run
   `ANALYZE <table>` and re-check. Still off? Skewed data / correlated columns
   (labs 02–03) or a function on the column hiding the value.
3. Check `BUFFERS: shared hit vs read`. All `hit` = cached. Lots of `read` on
   rerun = table bigger than RAM, or cache was dropped.
4. Check for `Sort` with `Sort Method: external merge  Disk:` — spilled
   (lab 05). Check for `Filter: ... Rows Removed by Filter: 199990` — you
   scanned everything to keep 10 rows; you probably want an index.

## 3. What the demo proves (all on one 200k-row table)

1. **Selective `WHERE` without index → Seq Scan** + `Rows Removed by Filter`.
   Same query **with index → Index/Bitmap Scan**, ~100× faster.
2. **`LIMIT` changes the plan.** `ORDER BY id LIMIT 10` wants an index scan
   (stop early); without the index it's Sort-of-200k then throw away.
3. **Join algorithm follows size.** Small selective probe → Nested Loop;
   full-table join → Hash Join. The demo forces both with `enable_*` flags so
   you see the shape, then lets the planner pick.
4. **Stale stats → wrong estimate.** Demo inserts a skewed batch, shows the
   estimate off by 100× *before* `ANALYZE`, correct *after*. Lesson: after
   bulk loads, `ANALYZE`.

## 4. Copy-paste recipes

```sql
EXPLAIN (ANALYZE, BUFFERS) SELECT ...;   -- the one command to memorize
ANALYZE my_table;                        -- rebuild stats after bulk load
SHOW work_mem;                           -- if you see disk sorts
SELECT * FROM pg_stat_user_tables        -- last_analyze, n_live_tup
WHERE relname = 'my_table';
```

## Exercises

1. In the demo's stale-stats step, note estimated vs actual rows before and
   after `ANALYZE`. What ratio would make you suspicious in production? (> 10×.)
2. Add `EXPLAIN` (no `ANALYZE`) for the selective query — compare its cost
   numbers with the `ANALYZE` runtime. Do higher-cost plans always run slower?
3. Run `SET enable_seqscan=off;` then re-`EXPLAIN` the full-count query. The
   planner obeys but the cost goes *up* — forcing indexes isn't a fix.
