# Lab 02 — Indexing fundamentals (B-tree, composite, covering)

**Question:** which index makes *this* query fast, and why was my index ignored?

**Answer:** a B-tree index lets Postgres jump to matching rows instead of
scanning. Three rules cover most real cases: (1) index the **selective**
column, (2) composite order matters — **leftmost prefix**, (3) a **covering**
index (`INCLUDE`) can answer without touching the table at all. Everything
else is the price: every index taxes writes and disk.

```bash
python3 demo.py              # default 200k rows
SCALE=1000000 python3 demo.py
```

## 1. When an index helps (selectivity)

- `WHERE status='priority'` matching 1% of 200k → index wins ~100×.
- `WHERE status='standard'` matching 99% → planner *correctly* Seq Scans;
  reading the whole table in order beats 198k random index hops.
- Rule of thumb: index lookups win below ~5–10% selectivity. Above that the
  planner switches to Seq Scan or Bitmap Scan — trust it (lab 01 showed how
  to verify with `BUFFERS`).

## 2. Composite order: leftmost prefix

`CREATE INDEX ON t (a, b)` serves `WHERE a=?`, `WHERE a=? AND b=?`, and
`ORDER BY a, b` — but **not** `WHERE b=?` alone (can't skip the first column,
like a phone book sorted by last,first can't find by first name).

So order = **equality columns first, then range/ORDER BY**, most-selective
equality first. The demo proves `(customer_id, status)` serves both the
single-column and two-column queries, while a lone `(status)` index can't
serve `WHERE customer_id=?`.

## 3. Covering indexes → Index Only Scans

```sql
CREATE INDEX ON orders (customer_id) INCLUDE (total);
SELECT total FROM orders WHERE customer_id = 42;
```

The leaf pages already hold `total`, so Postgres never reads the table
(`Heap Fetches: 0` in `EXPLAIN ANALYZE`). Needs `VACUUM` (visibility map) to
stay "only". Demo shows `BUFFERS shared hit` dropping ~10× vs plain index.
Cost: wider index = more disk + slower writes — only `INCLUDE` what hot
queries need.

## 4. Why your index was ignored (checklist)

1. **Function on the column:** `WHERE lower(email)=...` can't use a plain
   `(email)` index → expression index (lab 03).
2. **Leading wildcard:** `LIKE '%foo'` scans; `LIKE 'foo%'` can use B-tree.
3. **Type mismatch:** `WHERE text_col = 42` casts the column → no index.
   Cast the *parameter*, not the column.
4. **OR across columns:** `WHERE a=? OR b=?` usually scans; two indexes +
   `BitmapOr`, or a rewrite to `UNION`, fixes it (demo shows both plans).
5. **`IS NULL` / low selectivity / tiny table:** correct to ignore — a 50-row
   table is faster to scan, always.
6. **Stale stats:** estimate says "matches 90%" when it's really 1% → `ANALYZE`.

## 5. The price (demo measures it)

Bulk-insert timing with 0 vs 3 indexes: each extra B-tree adds write +
WAL + disk (`pg_relation_size`). That's why lab 07 hunts unused indexes.

## Exercises

1. Swap the composite to `(status, customer_id)` and re-run the
   `WHERE customer_id=?` query. Which plan? Why?
2. Add `WHERE total > 100` (range) to the composite query — does column order
   `(customer_id, total)` vs `(total, customer_id)` matter? Which serves
   equality + range best?
3. Run `SELECT pg_size_pretty(pg_relation_size('...'))` per index — what's the
   disk cost of `INCLUDE (total)` vs plain?
