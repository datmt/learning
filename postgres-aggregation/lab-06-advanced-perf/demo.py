"""Lab 06 — Ordered-set aggregates (median/p95/mode), list builders, COUNT DISTINCT, MV speed."""
import os
import random
import time
import psycopg2

CFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5433")),
    dbname=os.getenv("PGDATABASE", "reports"),
    user=os.getenv("PGUSER", "analyst"),
    password=os.getenv("PGPASSWORD", "analyst_pw"),
)
N = 50_000


def show(cur, title, sql, params=()):
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    print(f"\n-- {title}\n   {sql.strip()}")
    print("   | ".join(cols))
    for r in rows:
        print("   | ".join("NULL" if v is None else str(v) for v in r))
    return rows


def timed(cur, title, sql, params=()):
    t0 = time.perf_counter()
    cur.execute(sql, params)
    rows = cur.fetchall()
    ms = (time.perf_counter() - t0) * 1000
    print(f"\n-- {title}: {ms:.1f} ms, {len(rows)} group rows")
    for r in rows[:6]:
        print("   | ".join("NULL" if v is None else str(v) for v in r))
    return ms


def main():
    rnd = random.Random(7)
    regions = ["north", "south", "east", "west"]
    methods = ["card", "cash", "transfer"]
    rows = []
    for i in range(1, N + 1):
        # Skewed amounts: mostly small, occasional huge -> mean >> median.
        amount = round(min(rnd.expovariate(1 / 60) + 5, 5000), 2)
        day = (i % 28) + 1
        month = (i % 3) + 1  # Jan-Mar 2026
        rows.append((i, rnd.choice(regions), rnd.randint(1, 5000),
                     amount, rnd.choice(methods),
                     f"2026-{month:02d}-{day:02d} 10:00:00"))

    con = psycopg2.connect(**CFG)
    con.autocommit = True
    cur = con.cursor()
    cur.execute("DROP MATERIALIZED VIEW IF EXISTS mv_lab06_daily")
    cur.execute("DROP TABLE IF EXISTS lab06_payments")
    cur.execute("""CREATE TABLE lab06_payments (
        id INT PRIMARY KEY, region TEXT, customer INT, amount NUMERIC,
        method TEXT, ts TIMESTAMP)""")
    cur.executemany("INSERT INTO lab06_payments VALUES (%s,%s,%s,%s,%s,%s)", rows)
    print(f"seeded {N} rows into lab06_payments (seed=7)")

    show(cur, "1. Skewed data: mean misleads, median/p95 tell the truth",
         """SELECT region, COUNT(*) AS n,
                   ROUND(AVG(amount),2) AS mean,
                   ROUND(CAST(percentile_cont(0.5) WITHIN GROUP (ORDER BY amount)
                     AS numeric),2) AS median,
                   ROUND(CAST(percentile_cont(0.95) WITHIN GROUP (ORDER BY amount)
                     AS numeric),2) AS p95,
                   percentile_disc(0.9) WITHIN GROUP (ORDER BY amount) AS p90_actual_value,
                   mode() WITHIN GROUP (ORDER BY method) AS top_method
            FROM lab06_payments GROUP BY region ORDER BY region""")

    show(cur, "2. One value per group: method list + revenue map as JSON",
         """WITH rev AS (SELECT region, SUM(amount) AS revenue
                         FROM lab06_payments GROUP BY region)
            SELECT (SELECT string_agg(DISTINCT method, ',' ORDER BY method)
                    FROM lab06_payments) AS methods_seen,
                   (SELECT jsonb_object_agg(region, revenue) FROM rev) AS revenue_by_region""")

    t_all = timed(cur, "3a. COUNT(*) per region (cheap)",
                  "SELECT region, COUNT(*) FROM lab06_payments GROUP BY region ORDER BY 1")
    t_dist = timed(cur, "3b. COUNT(DISTINCT customer) per region (exact, costly)",
                   "SELECT region, COUNT(DISTINCT customer) FROM lab06_payments "
                   "GROUP BY region ORDER BY 1")
    print(f"   distinct-count overhead: {t_dist / max(t_all, 1e-9):.1f}x the plain count")

    print("\n-- 4. EXPLAIN baseline (seq scan + hash aggregate expected):")
    cur.execute("EXPLAIN SELECT region, SUM(amount) FROM lab06_payments GROUP BY region")
    for r in cur.fetchall():
        print("   " + r[0])
    cur.execute("CREATE INDEX IF NOT EXISTS ix_lab06_region ON lab06_payments(region)")
    cur.execute("ANALYZE lab06_payments")
    print("\n-- 5. EXPLAIN after index on grouping key + ANALYZE:")
    cur.execute("EXPLAIN SELECT region, SUM(amount) FROM lab06_payments GROUP BY region")
    for r in cur.fetchall():
        print("   " + r[0])

    cur.execute("""CREATE MATERIALIZED VIEW mv_lab06_daily AS
        SELECT date_trunc('day', ts)::date AS day, region,
               COUNT(*) AS n, SUM(amount) AS revenue,
               COUNT(DISTINCT customer) AS customers
        FROM lab06_payments GROUP BY 1, 2""")
    t0 = time.perf_counter()
    cur.execute("REFRESH MATERIALIZED VIEW mv_lab06_daily")
    print(f"\n-- 6. Materialized daily rollup refreshed in "
          f"{(time.perf_counter() - t0) * 1000:.1f} ms")
    timed(cur, "   dashboard query now reads the MV (tens of rows)",
          "SELECT region, SUM(revenue) FROM mv_lab06_daily GROUP BY region ORDER BY 1")
    print("\n-- EXPLAIN of the MV query:")
    cur.execute("EXPLAIN SELECT region, SUM(revenue) FROM mv_lab06_daily GROUP BY region")
    for r in cur.fetchall():
        print("   " + r[0])

    print("\nDone. Takeaway: percentile_cont/mode for distributions, *_agg for lists, "
          "COUNT DISTINCT is pricey, and pre-aggregate hot dashboards into a materialized view.")
    cur.close()
    con.close()


if __name__ == "__main__":
    main()
