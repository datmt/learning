# Lab 05 — Time-series reporting: buckets, gap filling, growth, cohorts

**Question:** my daily-revenue chart skips quiet days (line jumps instead of
touching zero), weekly numbers depend on "when a week starts", and the boss
wants retention by signup cohort. All SQL?

**Answer:** yes. `date_trunc` for bucketing, `generate_series` + `LEFT JOIN` for
gap filling, `LAG` for WoW/MoM, and a signup-month × activity-month matrix for cohorts.

```bash
python3 demo.py
```

Seeds (deterministic):
- `lab05_events` — 10 days of 2026-03-01…10 with **intentional gaps** (no sales
  on Mar 3, 6, 7) and a Monday-start week boundary inside.
- `lab05_purchases(user_id, bought_on)` + `lab05_users(user_id, signed_up_on)` —
  3 signup cohorts (Jan/Feb/Mar 2026) with decaying repeat purchases.

## 1. Bucketing with date_trunc

```sql
date_trunc('week', day)    -- Monday 00:00 (Postgres weeks start Monday)
date_trunc('month', day)   -- 1st of month
date_trunc('day', ts)      -- calendar day; use 'week'/'month' for coarser grains
```

`date_trunc` returns `timestamptz/date`-ish values — cast to `::date` for clean
report labels. Grouping by the raw timestamp would give one group per second:
bucket first, then group.

## 2. Gap filling — the chart fix

A plain `GROUP BY day` **omits** days with zero sales. The chart then connects
Mar-2 straight to Mar-4, hiding the dead day. Fix: build the full calendar with
`generate_series`, `LEFT JOIN` the data, `COALESCE` to 0:

```sql
WITH days AS (
  SELECT generate_series('2026-03-01'::date, '2026-03-10'::date, '1 day') AS day
)
SELECT d.day, COALESCE(SUM(e.revenue), 0) AS revenue
FROM days d LEFT JOIN lab05_events e ON e.day = d.day
GROUP BY d.day ORDER BY d.day;
-- Mar 3/6/7 now appear with 0 instead of vanishing.
```

This pattern (calendar CTE + LEFT JOIN) fixes every "missing periods" chart.

## 3. WoW growth on weekly buckets

Bucket to weeks, aggregate, then `LAG` over week order (lab 04 technique applied
to time). First week → NULL growth, correctly.

## 4. Cumulative revenue that respects gaps

`SUM(revenue) OVER (ORDER BY day ROWS UNBOUNDED PRECEDING)` on the *gap-filled*
series: flat segments on zero days. Run it on the unfilled series and the
x-axis itself is wrong.

## 5. Cohort retention matrix

Cohort = users sharing a signup month. For each cohort, what share bought again
in month 0 (signup month), 1, 2, ...?

```sql
-- month_index = months between signup and purchase
SELECT cohort_month,
       COUNT(DISTINCT CASE WHEN month_index = 0 THEN user_id END) AS m0,
       COUNT(DISTINCT CASE WHEN month_index = 1 THEN user_id END) AS m1, ...
```

(the `COUNT(DISTINCT CASE...)` idiom = conditional aggregation from lab 02 +
distinct counting). Divide by cohort size for retention rates. Demo prints raw
counts and the rate matrix — Jan cohort: 100% → 60% → 40%, Feb → …,
decaying as cohorts do.

## 6. What the demo proves

- Naive daily GROUP BY (7 rows, gaps hidden) vs gap-filled (10 rows with zeros).
- Weekly buckets + WoW growth via LAG.
- Cumulative curve over the filled calendar.
- Cohort size + absolute activity matrix + retention-rate matrix.

## Exercise

1. Change the week to start Sunday: `date_trunc('week', day + interval '1 day') - interval '1 day'`.
   Which daily rows move to a different week?
2. Extend retention to month 3 and add a `churned` column (in cohort, inactive
   all later months) — `EXCEPT` or `NOT EXISTS` against later activity.
