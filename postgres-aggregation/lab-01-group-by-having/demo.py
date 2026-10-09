"""Lab 01 — GROUP BY + HAVING. Seeds 12 hand-checkable rows, prints every claim."""
import os
import psycopg2

CFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5433")),
    dbname=os.getenv("PGDATABASE", "reports"),
    user=os.getenv("PGUSER", "analyst"),
    password=os.getenv("PGPASSWORD", "analyst_pw"),
)

# (id, customer, region, total, coupon) — NULLs placed deliberately.
SEED = [
    (1, "amy", "north", 100.0, "WELCOME"),
    (2, "amy", "north", 200.0, "WELCOME"),
    (3, "amy", "north", 50.0, None),
    (4, "amy", "north", 75.0, "SPRING"),
    (5, "amy", "north", 25.0, None),
    (6, "ben", "south", 300.0, "WELCOME"),
    (7, "ben", "south", 10.0, None),
    (8, "ben", "south", 20.0, "SPRING"),
    (9, "ben", "south", None, "SPRING"),   # price unknown -> NULL total
    (10, "cid", None, 500.0, "VIP"),       # unknown region -> NULL key
    (11, "cid", "north", 5.0, None),
    (12, "cid", "north", 8.0, "VIP"),
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
    cur.execute("DROP TABLE IF EXISTS lab01_orders")
    cur.execute("""CREATE TABLE lab01_orders (
        id INT PRIMARY KEY, customer TEXT, region TEXT,
        total NUMERIC, coupon TEXT)""")
    cur.executemany("INSERT INTO lab01_orders VALUES (%s,%s,%s,%s,%s)", SEED)
    print(f"seeded {len(SEED)} rows into lab01_orders")

    show(cur, "1. Per-customer totals (AVG ignores ben's NULL total)",
         """SELECT customer, COUNT(*) AS n, SUM(total) AS revenue,
                   ROUND(AVG(total),2) AS avg_order, MIN(total) AS min_o, MAX(total) AS max_o
            FROM lab01_orders GROUP BY customer ORDER BY customer""")

    show(cur, "2. COUNT(*) counts rows; COUNT(col) skips NULLs",
         "SELECT COUNT(*) AS rows_, COUNT(coupon) AS with_coupon, "
         "COUNT(DISTINCT coupon) AS distinct_coupons FROM lab01_orders")

    show(cur, "3. WHERE filters rows BEFORE grouping (cheap orders never counted)",
         """SELECT customer, COUNT(*) AS n, SUM(total) AS revenue
            FROM lab01_orders WHERE total > 20
            GROUP BY customer ORDER BY customer""")

    show(cur, "4. HAVING filters groups AFTER grouping (poor customers dropped)",
         """SELECT customer, COUNT(*) AS n, SUM(total) AS revenue
            FROM lab01_orders GROUP BY customer
            HAVING SUM(total) > 100 ORDER BY customer""")

    show(cur, "5. NULL region forms its own group",
         "SELECT region, COUNT(*) AS n, SUM(total) AS revenue "
         "FROM lab01_orders GROUP BY region ORDER BY revenue DESC NULLS LAST")

    show(cur, "6. Relabel it for dashboards with COALESCE in GROUP BY",
         """SELECT COALESCE(region,'unknown') AS region, COUNT(*) AS n, SUM(total) AS revenue
            FROM lab01_orders GROUP BY COALESCE(region,'unknown') ORDER BY revenue DESC""")

    show(cur, "7. HAVING on group size: customers with >= 4 orders",
         "SELECT customer, COUNT(*) AS n FROM lab01_orders "
         "GROUP BY customer HAVING COUNT(*) >= 4 ORDER BY customer")

    show(cur, "8. Group by expression: revenue per month",
         """SELECT date_trunc('month', d)::date AS month, COUNT(*) AS n, SUM(total) AS revenue
            FROM (SELECT total, date '2026-01-05' + (id || ' days')::interval AS d
                  FROM lab01_orders) s
            GROUP BY 1 ORDER BY 1""")

    print("\nDone. Takeaway: WHERE shapes the rows that get grouped; "
          "HAVING shapes which groups survive; aggregates skip NULLs (except COUNT(*)).")
    cur.close()
    con.close()


if __name__ == "__main__":
    main()
