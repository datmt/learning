"""Lab 04 — Window functions: ranking, running totals, MoM growth, % of total."""
import os
import psycopg2

CFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5433")),
    dbname=os.getenv("PGDATABASE", "reports"),
    user=os.getenv("PGUSER", "analyst"),
    password=os.getenv("PGPASSWORD", "analyst_pw"),
)

# Deliberate tie: gadget earns 300 in both Feb and Apr (shows RANK vs DENSE_RANK).
SEED = [
    ("gadget", "2026-01-01", 100),
    ("gadget", "2026-02-01", 300),
    ("gadget", "2026-03-01", 200),
    ("gadget", "2026-04-01", 300),
    ("gadget", "2026-05-01", 250),
    ("gadget", "2026-06-01", 400),
    ("widget", "2026-01-01", 500),
    ("widget", "2026-02-01", 450),
    ("widget", "2026-03-01", 600),
    ("widget", "2026-04-01", 550),
    ("widget", "2026-05-01", 700),
    ("widget", "2026-06-01", 650),
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
    cur.execute("DROP TABLE IF EXISTS lab04_monthly")
    cur.execute("CREATE TABLE lab04_monthly (product TEXT, month DATE, revenue NUMERIC)")
    cur.executemany("INSERT INTO lab04_monthly VALUES (%s,%s,%s)", SEED)
    print(f"seeded {len(SEED)} rows into lab04_monthly")

    show(cur, "1a. GROUP BY collapses: 2 rows, detail lost",
         "SELECT product, SUM(revenue) AS total FROM lab04_monthly "
         "GROUP BY product ORDER BY product")

    show(cur, "1b. Window annotates: 12 rows kept, total on each row",
         """SELECT product, month, revenue,
                   SUM(revenue) OVER (PARTITION BY product) AS product_total
            FROM lab04_monthly ORDER BY product, month""")

    show(cur, "2. The three rankings on data WITH a tie (gadget: two 300s)",
         """SELECT product, month, revenue,
                   ROW_NUMBER() OVER w AS row_number,
                   RANK() OVER w AS rank_,
                   DENSE_RANK() OVER w AS dense_rank
            FROM lab04_monthly
            WINDOW w AS (PARTITION BY product ORDER BY revenue DESC)
            ORDER BY product, rank_, month""")

    show(cur, "3. Top-2 months per product (CTE + RANK keeps ties -> gadget has 3 rows)",
         """WITH ranked AS (
              SELECT product, month, revenue,
                     RANK() OVER (PARTITION BY product ORDER BY revenue DESC) AS r
              FROM lab04_monthly)
            SELECT product, month, revenue FROM ranked
            WHERE r <= 2 ORDER BY product, revenue DESC""")

    show(cur, "4. Running total + 3-month moving average (explicit ROWS frames)",
         """SELECT product, month, revenue,
                   SUM(revenue) OVER (PARTITION BY product ORDER BY month
                     ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total,
                   ROUND(AVG(revenue) OVER (PARTITION BY product ORDER BY month
                     ROWS BETWEEN 2 PRECEDING AND CURRENT ROW), 2) AS moving_avg_3m
            FROM lab04_monthly ORDER BY product, month""")

    show(cur, "5. MoM growth via LAG (first month is NULL: no previous month)",
         """SELECT product, month, revenue,
                   LAG(revenue) OVER (PARTITION BY product ORDER BY month) AS prev,
                   ROUND((revenue - LAG(revenue) OVER (PARTITION BY product ORDER BY month))
                     / NULLIF(LAG(revenue) OVER (PARTITION BY product ORDER BY month), 0), 3)
                     AS mom_growth
            FROM lab04_monthly ORDER BY product, month""")

    show(cur, "6. Share of product, share of everything, NTILE quartiles",
         """SELECT product, month, revenue,
                   ROUND(revenue / SUM(revenue) OVER (PARTITION BY product), 3) AS share_of_product,
                   ROUND(revenue / SUM(revenue) OVER (), 3) AS share_of_all,
                   NTILE(4) OVER (ORDER BY revenue) AS revenue_quartile
            FROM lab04_monthly ORDER BY revenue DESC""")

    print("\nDone. Takeaway: windows annotate rows; GROUP BY collapses them. "
          "Rank in a CTE then filter; LAG/LEAD replace self-joins for period comparisons.")
    cur.close()
    con.close()


if __name__ == "__main__":
    main()
