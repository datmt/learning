"""Lab 03 — ROLLUP / CUBE / GROUPING SETS + GROUPING() labels. Seeds 24 rows."""
import os
import psycopg2

CFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5433")),
    dbname=os.getenv("PGDATABASE", "reports"),
    user=os.getenv("PGUSER", "analyst"),
    password=os.getenv("PGPASSWORD", "analyst_pw"),
)

# Fixed amounts: revenue = region_base + quarter_idx*10 + channel_bonus.
SEED = []
i = 0
for region, base in (("north", 100), ("south", 200)):
    for qi, quarter in enumerate(("Q1", "Q2", "Q3", "Q4")):
        for channel, bonus in (("web", 5), ("mobile", 15)):
            i += 1
            SEED.append((i, region, quarter, channel, base + qi * 10 + bonus))


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
    cur.execute("DROP TABLE IF EXISTS lab03_sales")
    cur.execute("""CREATE TABLE lab03_sales (
        id INT PRIMARY KEY, region TEXT, quarter TEXT, channel TEXT, amount NUMERIC)""")
    cur.executemany("INSERT INTO lab03_sales VALUES (%s,%s,%s,%s,%s)", SEED)
    print(f"seeded {len(SEED)} rows into lab03_sales")

    show(cur, "1. Plain GROUP BY: 8 detail rows, no subtotals",
         "SELECT region, quarter, SUM(amount) AS revenue FROM lab03_sales "
         "GROUP BY region, quarter ORDER BY region, quarter")

    rollup = show(cur, "2. ROLLUP: detail + per-region subtotal + grand total (11 rows)",
         "SELECT region, quarter, SUM(amount) AS revenue FROM lab03_sales "
         "GROUP BY ROLLUP (region, quarter) ORDER BY region NULLS LAST, quarter NULLS LAST")

    show(cur, "3. CUBE: also subtotals each quarter across regions (15 rows)",
         "SELECT region, quarter, SUM(amount) AS revenue FROM lab03_sales "
         "GROUP BY CUBE (region, quarter) ORDER BY region NULLS LAST, quarter NULLS LAST")

    show(cur, "4. GROUPING SETS: exactly the slices asked for, with readable labels",
         """SELECT CASE WHEN GROUPING(region)=1 THEN 'ALL regions' ELSE region END AS region,
                   CASE WHEN GROUPING(quarter)=1 THEN 'ALL quarters' ELSE quarter END AS quarter,
                   SUM(amount) AS revenue
            FROM lab03_sales
            GROUP BY GROUPING SETS ((region, quarter), (region), ())
            ORDER BY GROUPING(region), region, GROUPING(quarter), quarter""")

    manual = show(cur, "5. The manual UNION ALL way (same 11 rows as ROLLUP, 3 scans)",
         """SELECT region, quarter, SUM(amount) AS revenue FROM lab03_sales
              GROUP BY region, quarter
            UNION ALL
            SELECT region, NULL, SUM(amount) FROM lab03_sales GROUP BY region
            UNION ALL
            SELECT NULL, NULL, SUM(amount) FROM lab03_sales
            ORDER BY 1 NULLS LAST, 2 NULLS LAST""")
    assert len(manual) == len(rollup) == 11, (len(manual), len(rollup))
    print("\n   CHECK ok: UNION ALL rows == ROLLUP rows == 11, "
          f"grand total = {rollup[-1][2]}")

    print("\nDone. Takeaway: ROLLUP for hierarchies, CUBE for all combos (2-3 cols max), "
          "GROUPING SETS for exact slices; label with GROUPING().")
    cur.close()
    con.close()


if __name__ == "__main__":
    main()
