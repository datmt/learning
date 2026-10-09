# Lab 02 — Conditional aggregation & pivots: one scan, many metrics

**Question:** my dashboard needs one row per region with paid orders, refunded
orders, refund rate, web vs mobile revenue... Do I write 6 queries and join them?

**Answer:** no — aggregate `FILTER (WHERE ...)` or `CASE` inside the aggregate.
One table scan, one `GROUP BY`, as many metrics as you want. This is the single
most useful reporting technique in SQL.

```bash
python3 demo.py
```

Seed: `lab02_orders` — 2,000 deterministic orders (seed 42): region
north/south/east/west, channel web/mobile, status paid/refunded/pending, amount.

## 1. FILTER vs CASE — same result, different readability

```sql
-- FILTER: reads like English, Postgres-specific (standard SQL:2003, also in SQLite)
COUNT(*) FILTER (WHERE status = 'paid')           AS paid_orders,
SUM(amount) FILTER (WHERE channel = 'web')        AS web_revenue

-- CASE: portable to MySQL / BigQuery / every DB
COUNT(CASE WHEN status = 'paid' THEN 1 END)       AS paid_orders,
SUM(CASE WHEN channel = 'web' THEN amount END)    AS web_revenue
```

Why does the `CASE` version work? `CASE` returns NULL for non-matching rows, and
aggregates **ignore NULLs** (lab 01). `COUNT` of mostly-NULLs = count of matches.
`ELSE 0` variant: `SUM(CASE WHEN ... THEN amount ELSE 0 END)` — equivalent here
because 0 adds nothing, but `COUNT(CASE ... ELSE 0 END)` would be a **bug**
(0 is not NULL, so it gets counted!). The demo shows this trap.

## 2. Pivot: rows → columns

Raw data has one row per order with a `status`. The report wants one row per
region with a *column* per status. That's a pivot, done with conditional aggregation:

```sql
SELECT region,
       COUNT(*) FILTER (WHERE status = 'paid')     AS paid,
       COUNT(*) FILTER (WHERE status = 'refunded') AS refunded,
       COUNT(*) FILTER (WHERE status = 'pending')  AS pending
FROM lab02_orders GROUP BY region;
```

No `crosstab()` extension needed for a fixed, known set of columns — plain
`FILTER` is clearer and needs no extra install.

## 3. Ratios: the two classic bugs

```sql
-- BUG 1: integer division. COUNT returns bigint, so 3/4 = 0, not 0.75!
COUNT(*) FILTER (WHERE status='refunded') / COUNT(*)            -- always 0

-- FIX: cast one side
COUNT(*) FILTER (WHERE status='refunded')::float / COUNT(*)     AS refund_rate

-- BUG 2: division by zero when a group has no rows matching the denominator.
-- FIX: NULLIF turns 0 into NULL; x/NULL = NULL (blank in report, not an error)
SUM(...) / NULLIF(SUM(...), 0)
```

## 4. Boolean aggregates

`bool_and` / `bool_or` (a.k.a. `every`) answer "all / any" per group in one pass:
did *every* order in this region come from web? Did *any* get refunded?

## 5. What the demo proves

- One-row-per-region report with 8 metrics from a single scan.
- `FILTER` ≡ `CASE` (identical numbers), plus the `ELSE 0` COUNT trap.
- Status pivot (3 columns) + channel revenue pivot.
- Refund rate with `::float` + `NULLIF` guard; boolean `bool_and/bool_or` per region.

## Exercise

1. Add a column `paid_share_web`: share of *paid* revenue coming from web.
   (Hint: two FILTERs, one division, one NULLIF.)
2. Rewrite the pivot with `SUM((status='paid')::int)` — Postgres casts boolean to
   0/1. When is this trick handy, when is it obscure?
