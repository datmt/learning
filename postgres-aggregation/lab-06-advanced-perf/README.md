# Lab 06 — Ordered-set & misc aggregates + making big GROUP BYs fast

**Question:** median / p95 / most-common value in pure SQL? And my dashboard
`GROUP BY` over millions of rows takes seconds — what actually helps?

**Answer:** Postgres has ordered-set aggregates (`percentile_cont`, `mode()`)
plus `string_agg` / `array_agg` / `jsonb` builders for "list per group" reports.
For speed: filter early, index the grouping key, pre-aggregate (materialized
view), and check `EXPLAIN`.

```bash
python3 demo.py
```

Seed: `lab06_payments` — 50,000 deterministic rows (seed 7): region, customer
(5,000 distinct), amount (log-normal-ish: most small, few huge — so mean ≫
median and p95 matters), method card/cash/transfer.

## 1. Median / percentiles — ordered-set aggregates

```sql
percentile_cont(0.5)  WITHIN GROUP (ORDER BY amount) AS median,
percentile_cont(0.95) WITHIN GROUP (ORDER BY amount) AS p95,
percentile_disc(0.9)  WITHIN GROUP (ORDER BY amount) AS p90_discrete
```

- `percentile_cont` (continuous): interpolates — median of (10, 20) is 15.
- `percentile_disc` (discrete): returns an *actual* value from the group.
- `mode() WITHIN GROUP (ORDER BY method)`: most frequent value per group.
- Plain `AVG` on skewed data misleads: demo shows mean ~40% above median (64 vs ~46).
- Gotcha: `percentile_cont` returns `double precision`, and Postgres has no
  `round(float, n)` — cast first: `ROUND(CAST(percentile_cont(...) AS numeric), 2)`.

## 2. One value per group: string_agg / array_agg / jsonb

```sql
string_agg(DISTINCT method, ',' ORDER BY method)        AS methods_seen,
array_agg(amount ORDER BY amount DESC) ...              -- careful: huge arrays
jsonb_object_agg(region, revenue)                       -- build API-ready JSON
```

`ORDER BY` *inside* the aggregate controls list order. `DISTINCT` works inside
most aggregates. Warning: unbounded `array_agg`/`string_agg` on big groups eats
memory — cap with a `FILTER`, a `LIMIT` subquery, or aggregate daily first.

## 3. COUNT(DISTINCT) — exact but expensive

`COUNT(DISTINCT customer)` sorts/hashes all values per group — the costliest
common aggregate. Demo contrasts it with plain `COUNT(*)`: same groups, very
different timing. If exactness is optional at huge scale, extensions like
`t-digest`/`hyperloglog` give approximate distinct counts (mentioned, not installed).

## 4. Speed playbook (demo measures each)

| # | Technique | What changes |
|---|---|---|
| a | Baseline `GROUP BY region` over 50k rows | full seq scan + hash aggregate |
| b | `WHERE` on indexed column first | fewer rows into the aggregate |
| c | Index on the grouping key | measured: plan *stays* HashAggregate + Seq Scan — 4 distinct values sort for free in memory; the index doesn't pay off |
| d | Materialized view `mv_lab06_daily` | dashboard reads ~700 rows, not 50k (demo: 6.7 ms → 0.5 ms); `REFRESH MATERIALIZED VIEW` on schedule |

Bonus lesson from the measured plans: `ANALYZE` fixed the row estimate (200 → 4
groups) even though the strategy didn't change. Always `ANALYZE` after bulk loads.

`EXPLAIN (ANALYZE, BUFFERS)` shows the strategy: look for `HashAggregate` vs
`GroupAggregate`, `Seq Scan` vs `Index Scan`, and `rows=` estimates. The demo
prints plans for (a) and (d).

## 5. Pre-aggregation pattern for dashboards

```sql
CREATE MATERIALIZED VIEW mv_lab06_daily AS
  SELECT date_trunc('day', ts)::date AS day, region,
         COUNT(*), SUM(amount), COUNT(DISTINCT customer)
  FROM lab06_payments GROUP BY 1, 2;
-- dashboard: SELECT ... FROM mv_lab06_daily WHERE day BETWEEN ... (milliseconds)
-- refresh:  REFRESH MATERIALIZED VIEW CONCURRENTLY mv_lab06_daily; (needs unique index)
```

Roll up further (weekly from daily) instead of re-scanning raw rows.

## 6. What the demo proves

- mean vs median vs p95/p90/mode per region on skewed amounts.
- `string_agg` method list + `jsonb_object_agg` revenue map.
- `COUNT(*)` vs `COUNT(DISTINCT customer)` timings.
- `EXPLAIN` before index / after index / materialized-view query (ms).

## Exercise

1. Add `REFRESH MATERIALIZED VIEW` timing: how long is refresh vs how much does
   every dashboard query save? (Break-even math decides your refresh schedule.)
2. Run `EXPLAIN SELECT region ... GROUP BY region` before/after
   `CREATE INDEX ... ON lab06_payments(region)` — the plan *doesn't* change here.
   Why is an index on a 4-value grouping key useless, and what kind of `WHERE`
   clause *would* make an index pay off?
