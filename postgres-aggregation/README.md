# SQL Aggregation for Reporting — Hands-on Labs (PostgreSQL 16 + Docker)

Learn the SQL you actually use in reports and dashboards — not `SELECT ... WHERE ...`,
but **aggregation**: per-group totals, pivots, subtotals, running totals, time-series
with gaps filled, cohorts, and percentiles.

Each lab = `README.md` (theory, 5–10 min read) + `demo.py` (seeds its own data,
runs every query, prints proof). Labs are independent and self-seeding — run them
in any order, but 01 → 06 builds up.

## Setup (once)

Requirements: Docker + Python 3.10+.

```bash
cd postgres-aggregation

# 1. Start PostgreSQL 16 (port 5433, so it won't clash with a local 5432)
docker compose up -d
docker exec pg-aggregation-lab pg_isready -U analyst -d reports  # expect "accepting connections"

# 2. Python driver (only dependency)
pip install -r requirements.txt   # psycopg2-binary

# 3. Run a lab
cd lab-01-group-by-having && python3 demo.py
```

Connection defaults (override with env vars if you change the compose file):

| Var | Default |
|---|---|
| `PGHOST` | `localhost` |
| `PGPORT` | `5433` |
| `PGDATABASE` | `reports` |
| `PGUSER` | `analyst` |
| `PGPASSWORD` | `analyst_pw` |

Every `demo.py` connects with those vars, then `DROP TABLE IF EXISTS` + `CREATE TABLE`
+ deterministic seed, so re-running is always safe.

```bash
# Talk to the DB directly
docker exec -it pg-aggregation-lab psql -U analyst -d reports
# or from the host (needs local psql):
PGPASSWORD=analyst_pw psql -h localhost -p 5433 -U analyst -d reports
```

```bash
# Stop / wipe everything
docker compose down        # stop, keep data
docker compose down -v     # stop AND delete seeded data
```

## Lab map

| Lab | Question it answers | Key SQL |
|---|---|---|
| [01 GROUP BY + HAVING](lab-01-group-by-having/) | How do per-group totals work? Why is my `COUNT` wrong? `WHERE` vs `HAVING`? | `GROUP BY`, `COUNT(*)` vs `COUNT(col)`, `HAVING`, NULL groups |
| [02 conditional aggregation & pivots](lab-02-conditional-pivot/) | One row per X with many computed columns? Rows → columns without a BI tool? | `FILTER (WHERE ...)`, `CASE` inside aggregates, `NULLIF` div-by-zero guard |
| [03 subtotals: ROLLUP / CUBE / GROUPING SETS](lab-03-rollup-cube-grouping-sets/) | Subtotals + grand total in one query instead of 5 `UNION ALL`s? | `ROLLUP`, `CUBE`, `GROUPING SETS`, `GROUPING()` labels |
| [04 window functions for reporting](lab-04-window-reporting/) | Top-N per group, running totals, moving averages, MoM growth, % of total — without collapsing rows? | `OVER (PARTITION BY ... ORDER BY ...)`, `RANK`, `LAG/LEAD`, frames |
| [05 time-series, gaps & cohorts](lab-05-time-series-cohorts/) | Missing days disappear from my chart? WoW/MoM? Retention by signup cohort? | `date_trunc`, `generate_series` + `LEFT JOIN`, cohort matrix |
| [06 ordered-set aggregates & speed](lab-06-advanced-perf/) | Median / p95 / most-common value in SQL? Big `GROUP BY` slow? | `percentile_cont`, `mode()`, `string_agg`, `COUNT(DISTINCT)`, indexes, materialized views |

## Mental model (read once)

1. **Logical order is not written order.** `FROM → WHERE → GROUP BY → HAVING → SELECT → ORDER BY`.
   `WHERE` filters *rows before* grouping; `HAVING` filters *groups after* grouping.
2. **Aggregation collapses rows.** After `GROUP BY`, one output row per group. If you want to
   keep detail rows *and* add a total, that's a **window function** (lab 04), not `GROUP BY`.
3. **`NULL` is not a value.** Aggregates ignore `NULL`s (except `COUNT(*)`). A `NULL` group key
   forms its own group. `NULL = NULL` is never true — use `IS NULL` / `COALESCE`.
4. **One scan, many answers.** Conditional aggregation (`FILTER`, `CASE`) computes 10 metrics
   in one table pass — much faster than 10 queries or self-joins.
