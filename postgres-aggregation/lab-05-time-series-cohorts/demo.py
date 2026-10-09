"""Lab 05 — Time buckets, generate_series gap filling, WoW growth, cohort retention."""
import os
import psycopg2

CFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5433")),
    dbname=os.getenv("PGDATABASE", "reports"),
    user=os.getenv("PGUSER", "analyst"),
    password=os.getenv("PGPASSWORD", "analyst_pw"),
)

# 10 days with gaps on Mar 3, 6, 7 (no rows at all).
EVENTS = [
    ("2026-03-01", 100), ("2026-03-02", 200), ("2026-03-04", 150),
    ("2026-03-05", 300), ("2026-03-08", 250), ("2026-03-09", 100),
    ("2026-03-10", 400),
]

# Cohorts: (user, signup). Purchases decay over following months.
USERS = [
    (1, "2026-01-05"), (2, "2026-01-12"), (3, "2026-01-20"),
    (4, "2026-01-25"), (5, "2026-01-28"),
    (6, "2026-02-03"), (7, "2026-02-10"), (8, "2026-02-18"),
    (9, "2026-03-02"), (10, "2026-03-09"),
]
PURCHASES = [
    (1, "2026-01-05"), (1, "2026-02-06"), (1, "2026-03-07"),
    (2, "2026-01-12"), (2, "2026-02-15"),
    (3, "2026-01-20"),
    (4, "2026-01-25"), (4, "2026-03-02"),
    (5, "2026-01-28"), (5, "2026-02-02"),
    (6, "2026-02-03"), (6, "2026-03-05"),
    (7, "2026-02-10"),
    (8, "2026-02-18"), (8, "2026-03-20"),
    (9, "2026-03-02"),
    (10, "2026-03-09"),
]


def show(cur, title, sql, params=()):
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    print(f"\n-- {title}\n   {sql.strip()}")
    print("   | ".join(cols))
    for r in rows:
        print("   | ".join("NULL" if v is None else str(v) for v in r))
    return rows


def main():
    con = psycopg2.connect(**CFG)
    con.autocommit = True
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS lab05_events")
    cur.execute("CREATE TABLE lab05_events (day DATE, revenue NUMERIC)")
    cur.executemany("INSERT INTO lab05_events VALUES (%s,%s)", EVENTS)
    cur.execute("DROP TABLE IF EXISTS lab05_users")
    cur.execute("CREATE TABLE lab05_users (user_id INT PRIMARY KEY, signed_up_on DATE)")
    cur.executemany("INSERT INTO lab05_users VALUES (%s,%s)", USERS)
    cur.execute("DROP TABLE IF EXISTS lab05_purchases")
    cur.execute("CREATE TABLE lab05_purchases (user_id INT, bought_on DATE)")
    cur.executemany("INSERT INTO lab05_purchases VALUES (%s,%s)", PURCHASES)
    print(f"seeded {len(EVENTS)} event-days (gaps Mar 3/6/7), "
          f"{len(USERS)} users, {len(PURCHASES)} purchases")

    show(cur, "1. NAIVE daily GROUP BY: gaps vanish (7 rows, chart lies)",
         "SELECT day, SUM(revenue) AS revenue FROM lab05_events "
         "GROUP BY day ORDER BY day")

    show(cur, "2. Gap-filled via generate_series + LEFT JOIN (10 rows, zeros visible)",
         """WITH days AS (
              SELECT generate_series('2026-03-01'::date, '2026-03-10'::date, '1 day')::date AS day
            )
            SELECT d.day, COALESCE(SUM(e.revenue), 0) AS revenue
            FROM days d LEFT JOIN lab05_events e ON e.day = d.day
            GROUP BY d.day ORDER BY d.day""")

    show(cur, "3. Weekly buckets (weeks start Monday) + WoW growth",
         """WITH weeks AS (
              SELECT date_trunc('week', day)::date AS week, SUM(revenue) AS revenue
              FROM lab05_events GROUP BY 1)
            SELECT week, revenue,
                   LAG(revenue) OVER (ORDER BY week) AS prev_week,
                   ROUND((revenue - LAG(revenue) OVER (ORDER BY week))
                     / NULLIF(LAG(revenue) OVER (ORDER BY week), 0), 3) AS wow_growth
            FROM weeks ORDER BY week""")

    show(cur, "4. Cumulative revenue over the FILLED calendar (flat on zero days)",
         """WITH days AS (
              SELECT generate_series('2026-03-01'::date, '2026-03-10'::date, '1 day')::date AS day)
            SELECT d.day, COALESCE(SUM(e.revenue),0) AS revenue,
                   SUM(COALESCE(SUM(e.revenue),0)) OVER (ORDER BY d.day
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS cumulative
            FROM days d LEFT JOIN lab05_events e ON e.day = d.day
            GROUP BY d.day ORDER BY d.day""")

    show(cur, "5. Cohort sizes + active users per month-index (conditional COUNT DISTINCT)",
         """WITH act AS (
              SELECT u.user_id,
                     date_trunc('month', u.signed_up_on)::date AS cohort,
                     (EXTRACT(YEAR FROM p.bought_on)*12 + EXTRACT(MONTH FROM p.bought_on))
                     - (EXTRACT(YEAR FROM u.signed_up_on)*12 + EXTRACT(MONTH FROM u.signed_up_on)) AS m
              FROM lab05_users u JOIN lab05_purchases p USING (user_id))
            SELECT cohort, COUNT(DISTINCT user_id) AS cohort_size,
                   COUNT(DISTINCT CASE WHEN m=0 THEN user_id END) AS m0,
                   COUNT(DISTINCT CASE WHEN m=1 THEN user_id END) AS m1,
                   COUNT(DISTINCT CASE WHEN m=2 THEN user_id END) AS m2
            FROM act GROUP BY cohort ORDER BY cohort""")

    show(cur, "6. Retention-rate matrix (share of cohort active in m0/m1/m2)",
         """WITH act AS (
              SELECT u.user_id,
                     date_trunc('month', u.signed_up_on)::date AS cohort,
                     (EXTRACT(YEAR FROM p.bought_on)*12 + EXTRACT(MONTH FROM p.bought_on))
                     - (EXTRACT(YEAR FROM u.signed_up_on)*12 + EXTRACT(MONTH FROM u.signed_up_on)) AS m
              FROM lab05_users u JOIN lab05_purchases p USING (user_id))
            SELECT cohort,
              ROUND(COUNT(DISTINCT CASE WHEN m=0 THEN user_id END)::numeric
                / COUNT(DISTINCT user_id), 2) AS m0_rate,
              ROUND(COUNT(DISTINCT CASE WHEN m=1 THEN user_id END)::numeric
                / COUNT(DISTINCT user_id), 2) AS m1_rate,
              ROUND(COUNT(DISTINCT CASE WHEN m=2 THEN user_id END)::numeric
                / COUNT(DISTINCT user_id), 2) AS m2_rate
            FROM act GROUP BY cohort ORDER BY cohort""")

    print("\nDone. Takeaway: bucket with date_trunc, fill gaps with generate_series + "
          "LEFT JOIN, grow with LAG, retain with cohort x month-index matrix.")
    cur.close()
    con.close()


if __name__ == "__main__":
    main()
