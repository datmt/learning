# Lab 03 — Subtotals in one query: ROLLUP, CUBE, GROUPING SETS

**Question:** my report needs detail rows *plus* subtotals per region *plus* a
grand total. Do I run 3 queries and `UNION ALL` them?

**Answer:** `ROLLUP` / `CUBE` / `GROUPING SETS` do it in one pass, and `GROUPING()`
tells subtotal rows apart from real `NULL`s.

```bash
python3 demo.py
```

Seed: `lab03_sales` — 16 rows, every combination of region (north/south) ×
quarter (Q1–Q4) × channel (web/mobile), fixed amounts so you can add them up by hand.

## 1. The manual way (what these replace)

```sql
SELECT region, quarter, SUM(amount) FROM lab03_sales GROUP BY region, quarter
UNION ALL
SELECT region, NULL, SUM(amount) FROM lab03_sales GROUP BY region   -- subtotal
UNION ALL
SELECT NULL, NULL, SUM(amount) FROM lab03_sales;                    -- grand total
-- 3 scans, verbose, easy to get ORDER BY wrong.
```

## 2. ROLLUP — hierarchy: detail → subtotal → grand total

```sql
SELECT region, quarter, SUM(amount) AS revenue
FROM lab03_sales
GROUP BY ROLLUP (region, quarter);
```

Produces, per region: 4 quarter rows + 1 region subtotal (quarter = NULL) — plus
1 grand total (both NULL). `ROLLUP (a, b, c)` gives grouping sets
`(a,b,c) / (a,b) / (a) / ()`. Use it when columns form a hierarchy
(year → quarter → month, country → city).

## 3. CUBE — every combination

`CUBE (region, quarter)` adds the *other* direction too: subtotals per quarter
across regions. Grouping sets: `(r,q) / (r) / (q) / ()`. Cost: 2ⁿ sets for n
columns — fine for 2–3 columns, explosive beyond that. For 4+ dimensions, spell
out exactly what you need with `GROUPING SETS`.

## 4. GROUPING SETS — pick exactly the slices you want

```sql
GROUP BY GROUPING SETS ((region, quarter), (region), ())
-- = ROLLUP(region, quarter), but you control the list.
```

Common reporting pattern: `(region, quarter)` detail + `(region)` subtotal +
`()` grand total, but *not* the `(quarter)` slice nobody asked for.

## 5. GROUPING() — is this NULL a subtotal or real data?

Subtotal rows show NULL in the rolled-up column — indistinguishable from a genuine
NULL value (lab 01: our NULL-region row). `GROUPING(col)` returns 1 on subtotal/
total rows, 0 otherwise. Standard labeling pattern:

```sql
SELECT CASE WHEN GROUPING(region) = 1 THEN 'ALL regions' ELSE region END AS region,
       CASE WHEN GROUPING(quarter) = 1 THEN 'ALL quarters' ELSE quarter END AS quarter,
       SUM(amount) AS revenue
FROM lab03_sales GROUP BY ROLLUP (region, quarter);
```

`GROUPING_ID` / bitmask variants exist; per-column `GROUPING()` + `CASE` is the
readable default.

## 6. What the demo proves

- `ROLLUP (region, quarter)`: 8 detail + 2 subtotal + 1 grand total = 11 rows.
- `CUBE`: adds 4 quarter-subtotals = 15 rows.
- Custom `GROUPING SETS` with human labels via `GROUPING()`.
- Equivalence check: manual `UNION ALL` row count = `ROLLUP` row count.

## Exercise

1. Write `ROLLUP (quarter, region)` — how does the row list differ from
   `ROLLUP (region, quarter)`? (Column order = hierarchy order.)
2. Add a `GROUPING SETS` query giving detail + grand total only (no subtotals).
   When is that the right report?
