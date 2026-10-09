"""Lab 01 — read EXPLAIN output: scans, limits, joins, stale stats."""
import os
import psycopg2

CFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5434")),
    dbname=os.getenv("PGDATABASE", "perf"),
    user=os.getenv("PGUSER", "perf"),
    password=os.getenv("PGPASSWORD", "perf_pw"),
)
SCALE = int(os.getenv("SCALE", "200000"))


def explain(cur, title, sql, note=""):
    cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, TIMING) {sql}")
    print(f"\n-- {title}" + (f"  [{note}]" if note else ""))
    for (line,) in cur.fetchall():
        print(f"   {line}")


def main():
    con = psycopg2.connect(**CFG)
    con.autocommit = True
    cur = con.cursor()

    print(f"== seed {SCALE} orders + 5k customers ==")
    cur.execute("DROP TABLE IF EXISTS lab01_orders")
    cur.execute("DROP TABLE IF EXISTS lab01_customers")
    cur.execute("""CREATE TABLE lab01_customers (
        id INT PRIMARY KEY, name TEXT, region TEXT)""")
    cur.execute("""CREATE TABLE lab01_orders (
        id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
        customer_id INT NOT NULL, status TEXT NOT NULL,
        total NUMERIC NOT NULL, placed_at TIMESTAMPTZ NOT NULL)""")
    cur.execute("INSERT INTO lab01_customers SELECT g, 'c'||g, "
                "CASE WHEN mod(g, 4) = 0 THEN 'south' ELSE 'north' END "
                "FROM generate_series(1, 5000) g")
    # 1% 'priority', 99% 'standard' — selective value to hunt for
    cur.execute(
        """INSERT INTO lab01_orders (customer_id, status, total, placed_at)
           SELECT 1 + mod(g, 5000),
                  CASE WHEN mod(g, 100) = 0 THEN 'priority' ELSE 'standard' END,
                  mod(g, 500) + 1, now() - (g || ' seconds')::interval
           FROM generate_series(1, %s) g""", (SCALE,))
    cur.execute("VACUUM ANALYZE lab01_orders")
    cur.execute("VACUUM ANALYZE lab01_customers")
    print(f"   seeded {SCALE} orders; 1% priority (~{SCALE//100} rows)")

    print("\n== 1. selective WHERE without index -> Seq Scan + Rows Removed ==")
    explain(cur, "no index: status='priority'",
            "SELECT count(*) FROM lab01_orders WHERE status='priority'")

    print("\n== 2. same query with index -> Bitmap/Index Scan ==")
    cur.execute("CREATE INDEX lab01_orders_status ON lab01_orders(status)")
    cur.execute("VACUUM ANALYZE lab01_orders")
    explain(cur, "with index: status='priority'",
            "SELECT count(*) FROM lab01_orders WHERE status='priority'")

    print("\n== 3. LIMIT wants to stop early ==")
    explain(cur, "ORDER BY id LIMIT 10 (index lets it stop after 10)",
            "SELECT id, total FROM lab01_orders ORDER BY id LIMIT 10")

    print("\n== 4. join algorithm follows size ==")
    explain(cur, "selective probe -> Nested Loop (10 orders x PK lookup)",
            """SELECT o.id, c.name FROM lab01_orders o
               JOIN lab01_customers c ON c.id = o.customer_id
               WHERE o.status='priority' LIMIT 10""")
    explain(cur, "full join -> Hash Join (all rows both sides)",
            """SELECT count(*) FROM lab01_orders o
               JOIN lab01_customers c ON c.id = o.customer_id""")

    print("\n== 5. stale stats -> misestimate, ANALYZE fixes it ==")
    cur.execute("INSERT INTO lab01_orders (customer_id, status, total, placed_at) "
                "SELECT 1, 'flashsale', 5, now() FROM generate_series(1, 20000) g")
    # NOTE: deliberately no ANALYZE yet
    explain(cur, "before ANALYZE (est rows will be far off)",
            "SELECT count(*) FROM lab01_orders WHERE status='flashsale'",
            note="compare rows est vs actual")
    cur.execute("ANALYZE lab01_orders")
    explain(cur, "after ANALYZE (est should match actual ~20000)",
            "SELECT count(*) FROM lab01_orders WHERE status='flashsale'",
            note="lesson: ANALYZE after bulk loads")

    cur.execute("DROP INDEX IF EXISTS lab01_orders_status")
    print("\nDone. Takeaway: est-rows vs actual-rows is the whole game; "
          "LIMIT/join-size steer the plan; ANALYZE after bulk writes.")
    cur.close()
    con.close()


if __name__ == "__main__":
    main()
