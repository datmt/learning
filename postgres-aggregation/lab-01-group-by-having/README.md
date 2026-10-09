# Lab 01 — GROUP BY and HAVING: per-group totals done right

**Question:** how do I get one row per customer / day / status with totals, and why
are my counts sometimes wrong?

**Answer:** `GROUP BY` collapses rows into one row per group. `WHERE` filters rows
*before* grouping, `HAVING` filters groups *after*. Aggregates ignore `NULL`,
except `COUNT(*)`. The demo seeds a tiny 12-row `lab01_orders` table you can verify by hand.

```bash
python3 demo.py   # needs the DB from the top-level README (docker compose up -d)
```

## 1. Logical query order (the #1 source of confusion)

```sql
FROM lab01_orders
WHERE  ...    -- 1st: throw away rows (e.g. only 2026, only 'paid')
GROUP BY ...  -- 2nd: bucket remaining rows into groups
HAVING ...    -- 3rd: throw away whole groups (e.g. groups with < 2 orders)
SELECT ...    -- 4th: one output row per surviving group
ORDER BY ...  -- 5th: sort the groups
```

So `WHERE total > 100` means "ignore cheap *orders*", while
`HAVING SUM(total) > 100` means "ignore poor *customers*". Different things!

## 2. COUNT variants — they differ when NULLs exist

| Expression | Counts | Ignores NULL? |
|---|---|---|
| `COUNT(*)` | rows | never NULL — counts everything |
| `COUNT(col)` | non-NULL values in `col` | yes |
| `COUNT(DISTINCT col)` | distinct non-NULL values | yes |

Our seed has an order with `coupon = NULL` (no coupon used). `COUNT(*)` = 12,
`COUNT(coupon)` = 8. If you want "share of orders with a coupon", that's
`COUNT(coupon) / COUNT(*)::float` — and the `::float` matters (see lab 02's
integer-division trap: `8/12 = 0` in Postgres without a cast).

## 3. AVG / SUM ignore NULLs too

`AVG(total)` divides by the number of *non-NULL* totals, not by `COUNT(*)`.
If a `total` is NULL (e.g. pending order, price unknown), it simply doesn't
participate. `SUM` of all-NULL group → NULL, not 0 — wrap with
`COALESCE(SUM(x), 0)` for dashboards.

## 4. NULL group keys form their own group

One seed row has `region = NULL` (unknown). `GROUP BY region` produces a group
with a NULL key. In dashboards, relabel it: `COALESCE(region, 'unknown')`.
Note: you must repeat the expression — `GROUP BY COALESCE(region,'unknown')` —
or group by ordinal / use a subquery, because `SELECT` aliases can't be used
in `GROUP BY`... actually in Postgres they **can** (`GROUP BY` runs before
`SELECT` logically, but Postgres lets you reference output aliases — a handy
extension the demo uses).

## 5. Grouping by expression

`GROUP BY date_trunc('month', placed_at)` groups by month. Postgres lets you
`GROUP BY 1` (first select item) — convenient, brittle if you reorder columns.
Prefer repeating the expression or alias.

## 6. What the demo proves (expected output)

Seed: 12 orders, 3 customers (amy ×5, ben ×4, cid ×3), one NULL region, one
NULL total, three NULL coupons.

- per-customer `COUNT(*)`, `SUM`, `AVG`, `MIN`, `MAX` — ben's AVG ignores his NULL total.
- `COUNT(*)`=12 vs `COUNT(coupon)`=9.
- `WHERE` (cheap orders removed first) vs `HAVING` (poor customers removed after) side by side.
- NULL region group + `COALESCE` relabel.
- `HAVING COUNT(*) >= 4` keeps only amy & ben.

## Exercise

1. Write "revenue per region, only regions with revenue > 200, including the
   `unknown` bucket" — where does `COALESCE` go, `WHERE` or `GROUP BY`?
2. Predict, then check: `SELECT AVG(total) FROM lab01_orders;` — divisor is 12 or 11? Why?
