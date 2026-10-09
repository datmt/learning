# Lab 04 — Window functions: totals without collapsing rows

**Question:** `GROUP BY` collapses rows — but my report needs *each* order plus
its running total, or the top 3 products *per category*, or each month next to
its growth vs last month. Self-joins?

**Answer:** window functions compute over a "window" of related rows while
**keeping every row**. Syntax: `FUNC(...) OVER (PARTITION BY ... ORDER BY ... <frame>)`.

```bash
python3 demo.py
```

Seed: `lab04_monthly` — 2 products (gadget, widget) × 6 months (2026-01…06),
fixed revenues with a dip, so running totals / growth are hand-checkable.

## 1. GROUP BY vs window — the core distinction

```sql
-- GROUP BY: 2 rows (one per product). Detail gone.
SELECT product, SUM(revenue) FROM lab04_monthly GROUP BY product;

-- Window: 12 rows (detail kept), total attached to each row.
SELECT product, month, revenue,
       SUM(revenue) OVER (PARTITION BY product) AS product_total
FROM lab04_monthly;
```

Rule of thumb: collapsing → `GROUP BY`; annotating → window.

## 2. Ranking: ROW_NUMBER vs RANK vs DENSE_RANK

Top-N **per group** = rank inside `PARTITION BY product`, then filter.
They differ on ties (two months with revenue 300):

| Function | Tied rows get | Next row gets | Use when |
|---|---|---|---|
| `ROW_NUMBER()` | 1, 2 (arbitrary order) | 3 | need exactly N rows |
| `RANK()` | 1, 1 | 3 (gap) | competition ranking ("3rd place" skips) |
| `DENSE_RANK()` | 1, 1 | 2 (no gap) | top-N distinct levels |

Gotcha: you can't put a window function in `WHERE` (windows run *after*
`WHERE`). Wrap in a subquery/CTE, then filter — the demo does exactly this.

## 3. Running total & moving average — frames

`ORDER BY` inside `OVER` defines row order; the **frame** defines which rows count:

```sql
SUM(revenue) OVER (PARTITION BY product ORDER BY month
                   ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW)  -- running total
AVG(revenue) OVER (PARTITION BY product ORDER BY month
                   ROWS BETWEEN 2 PRECEDING AND CURRENT ROW)          -- 3-month moving avg
```

Default frame with `ORDER BY` is `RANGE ... CURRENT ROW` (peers included) —
for unique month orderings `ROWS` and `RANGE` agree, but with ties they differ.
Being explicit with `ROWS` avoids surprises.

## 4. LAG / LEAD — month-over-month growth without a self-join

```sql
LAG(revenue) OVER (PARTITION BY product ORDER BY month) AS prev_month,
(revenue - LAG(revenue) OVER (...)) / NULLIF(LAG(revenue) OVER (...), 0) AS mom_growth
```

First row's `LAG` is NULL → growth NULL. That's correct (no previous month),
not an error. `NULLIF(..., 0)` guards a zero previous month. `LEAD` looks
forward ("next month"). (No `::float` cast needed here: `revenue` is `NUMERIC`,
so division is already exact — the cast matters for integer division, lab 02.)

## 5. Percent of total in one query

```sql
revenue / SUM(revenue) OVER (PARTITION BY product) AS share_of_product,
revenue / SUM(revenue) OVER ()                     AS share_of_all
```

`OVER ()` = whole result set as one window. No `GROUP BY`, no join.

## 6. What the demo proves

- GROUP BY (2 rows) vs window total (12 annotated rows) side by side.
- All three rankings on data with a tie — see the gap / no-gap behavior.
- Top-2 months per product via CTE + `RANK() <= 2` (keeps ties — 3 rows for gadget).
- Running total, 3-month moving average, MoM growth with NULL first month.
- Share-of-product and share-of-total; `NTILE(4)` quartiles.

## Exercise

1. Change the top-N query to `ROW_NUMBER() <= 2`. How many rows now, and why is
   that sometimes *wrong* for a "top 2" report with ties?
2. Write quarter-to-date revenue: `SUM(revenue) OVER (PARTITION BY product,
   date_trunc('quarter', month) ORDER BY month)`.
