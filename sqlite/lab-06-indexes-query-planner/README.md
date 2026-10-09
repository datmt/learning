# Lab 6 — Indexes and the Query Planner

**Question:** my query is slow. How do I know if SQLite uses my index?

**Answer:** prefix the query with `EXPLAIN QUERY PLAN`. Look for `SCAN` (reads everything)
vs `SEARCH` (jumps via an index). The demo builds a 500k-row `orders` table and tries 9 scenarios.

```bash
python3 demo.py
```

## 1. Reading a plan

| Plan text | Meaning |
|---|---|
| `SCAN orders` | full table read, O(n) |
| `SEARCH orders USING INDEX ix (col=?)` | B-tree seek in index, then fetch row from table |
| `USING COVERING INDEX` | everything needed is in the index → no table fetch |
| `USING INTEGER PRIMARY KEY (rowid=?)` | fastest: direct seek in the table's own B-tree |
| `SCAN ... USING COVERING INDEX` | still a full scan, just over the (smaller) index |
| `USE TEMP B-TREE FOR ORDER BY` | had to sort in a temp structure |
| `MULTI-INDEX OR` | two index searches, results merged |

In the `sqlite3` CLI: `.eqp on` prints the plan for every query.

## 2. Results (sample run)

| # | Scenario | Plan | Time |
|---|---|---|---|
| 1 | `customer_id = ?`, no index | SCAN | 10 ms |
| 1 | with index | SEARCH | 0.01 ms (**1000x**) |
| 2 | composite `(status, created_at)`, filter on `created_at` only | SCAN (index) | 7.5 ms |
| 3 | range query, plain index vs covering index | SEARCH vs COVERING | 14 ms → 4.7 ms |
| 4 | `lower(email) = ?` with index on `email` | SCAN | 23 ms |
| 4 | expression index on `lower(email)` | SEARCH | 0.003 ms |
| 5 | `CAST(code AS INT) = 123` | SCAN | 2 ms |
| 6 | partial index `WHERE status='pending'` | 33 pages vs 1588 for full index | |
| 7 | `LIKE 'User42@%'` | SCAN | 12 ms |
| 7 | `GLOB 'User42@*'` | SEARCH | 0.002 ms |
| 8 | `ORDER BY total DESC LIMIT 10`, no index | TEMP B-TREE sort | 12 ms |
| 8 | with index on `total` | index order, stop after 10 | 0.006 ms |
| 9 | low-selectivity index, before vs after `ANALYZE` | SEARCH ix_status → SCAN | 22 ms → 15 ms |

## 3. Rules learned

1. **Leftmost prefix:** index `(a, b, c)` helps `a`, `a,b`, `a,b,c` — not `b` alone.
   Put equality columns first, the range column last.
2. **Covering index:** add the selected columns to the index to skip table lookups.
3. **Don't wrap indexed columns in functions/CAST.** Index the expression instead:
   `CREATE INDEX ... ON t(lower(email))`. The query must use the *same* expression.
4. **Partial index** (`... WHERE status = 'pending'`): tiny index for a hot subset. Only used when the
   query's `WHERE` clearly implies the index's `WHERE`.
5. **`LIKE` can't use a normal index** because it's case-insensitive. Options: `GLOB 'x*'`,
   an index with `COLLATE NOCASE`, or `PRAGMA case_sensitive_like = ON`. Leading `%` never uses an index → use FTS5 (lab 7).
6. **Index the ORDER BY** for "top N" queries.
7. **Run `ANALYZE`** (or `PRAGMA optimize`). Without stats the planner assumes every index is selective.
   Here, `status` has 3 values (166,667 rows each): reading via that index is *slower* than a scan.
8. **SQLite mostly uses one index per table per query** (except `OR`).

## 4. Cost of indexes

Every index = another B-tree to update on each INSERT/UPDATE/DELETE + disk space.
Find unused ones by checking plans of your real queries. Don't index columns with few distinct values.

## Production tips

- Run `PRAGMA optimize;` before closing long-lived connections (or periodically). It runs `ANALYZE` only where needed.
- Log slow queries in your app and check their `EXPLAIN QUERY PLAN`.
- `sqlite3 app.db ".expert"` then a query → the CLI suggests indexes.

## Exercise

1. Write a query on `orders` that uses `ix_cust_total` as a covering index *and* returns rows ordered by `total`. Is a temp B-tree needed?
2. Create `CREATE INDEX ix_e_nocase ON orders(email COLLATE NOCASE)` and test `LIKE 'User42@%'` again.
