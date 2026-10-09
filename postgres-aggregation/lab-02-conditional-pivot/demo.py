"""Lab 02 — Conditional aggregation (FILTER / CASE), pivots, ratios. Seeds 2000 rows."""
import os
import random
import psycopg2

CFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5433")),
    dbname=os.getenv("PGDATABASE", "reports"),
    user=os.getenv("PGUSER", "analyst"),
    password=os.getenv("PGPASSWORD", "analyst_pw"),
)


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
    rnd = random.Random(42)
    regions = ["north", "south", "east", "west"]
    rows = []
    for i in range(1, 2001):
        region = rnd.choice(regions)
        channel = "web" if rnd.random() < 0.6 else "mobile"
        u = rnd.random()
        status = "paid" if u < 0.80 else ("refunded" if u < 0.90 else "pending")
        amount = round(rnd.uniform(5, 500), 2)
        rows.append((i, region, channel, status, amount))

    con = psycopg2.connect(**CFG)
    con.autocommit = True
    cur = con.cursor()
    cur.execute("DROP TABLE IF EXISTS lab02_orders")
    cur.execute("""CREATE TABLE lab02_orders (
        id INT PRIMARY KEY, region TEXT, channel TEXT, status TEXT, amount NUMERIC)""")
    cur.executemany("INSERT INTO lab02_orders VALUES (%s,%s,%s,%s,%s)", rows)
    print(f"seeded {len(rows)} rows into lab02_orders (seed=42)")

    show(cur, "1. One row per region, 8 metrics, ONE scan (FILTER)",
         """SELECT region,
                   COUNT(*) AS orders,
                   COUNT(*) FILTER (WHERE status='paid') AS paid,
                   COUNT(*) FILTER (WHERE status='refunded') AS refunded,
                   SUM(amount) FILTER (WHERE status='paid') AS paid_revenue,
                   SUM(amount) FILTER (WHERE channel='web') AS web_revenue,
                   SUM(amount) FILTER (WHERE channel='mobile') AS mobile_revenue,
                   ROUND(AVG(amount) FILTER (WHERE status='paid'), 2) AS avg_paid
            FROM lab02_orders GROUP BY region ORDER BY region""")

    show(cur, "2. Same numbers with portable CASE (NULLs are ignored by aggregates)",
         """SELECT region,
                   COUNT(CASE WHEN status='paid' THEN 1 END) AS paid,
                   SUM(CASE WHEN channel='web' THEN amount END) AS web_revenue
            FROM lab02_orders GROUP BY region ORDER BY region""")

    show(cur, "3. TRAP: ELSE 0 inside COUNT counts everything (0 is not NULL)",
         """SELECT COUNT(CASE WHEN status='paid' THEN 1 END) AS correct,
                   COUNT(CASE WHEN status='paid' THEN 1 ELSE 0 END) AS buggy_always_2000
            FROM lab02_orders""")

    show(cur, "4. Pivot: order statuses become columns",
         """SELECT region,
                   COUNT(*) FILTER (WHERE status='paid') AS paid,
                   COUNT(*) FILTER (WHERE status='refunded') AS refunded,
                   COUNT(*) FILTER (WHERE status='pending') AS pending
            FROM lab02_orders GROUP BY region ORDER BY region""")

    show(cur, "5. Ratios: cast to float + NULLIF div-by-zero guard",
         """SELECT region,
                   COUNT(*) FILTER (WHERE status='refunded')::float / COUNT(*) AS refund_rate,
                   SUM(amount) FILTER (WHERE channel='web')
                     / NULLIF(SUM(amount) FILTER (WHERE status='paid'), 0) AS web_share_of_paid
            FROM lab02_orders GROUP BY region ORDER BY region""")

    show(cur, "6. Boolean aggregates: all-web? any refund?",
         """SELECT region,
                   bool_and(channel='web') AS all_web,
                   bool_or(status='refunded') AS any_refund
            FROM lab02_orders GROUP BY region ORDER BY region""")

    print("\nDone. Takeaway: FILTER/CASE = many metrics per group in one pass; "
          "always cast ratios to float and guard with NULLIF.")
    cur.close()
    con.close()


if __name__ == "__main__":
    main()
