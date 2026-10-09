"""Lab 05 — join algorithms, work_mem spill, N+1, EXISTS vs IN, CTE materialization."""
import os
import time
import psycopg2

CFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5434")),
    dbname=os.getenv("PGDATABASE", "perf"),
    user=os.getenv("PGUSER", "perf"),
    password=os.getenv("PGPASSWORD", "perf_pw"),
)
SCALE = int(os.getenv("SCALE", "300000"))


def explain(cur, title, sql):
    cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, TIMING) {sql}")
    print(f"\n-- {title}\n   {sql[:160]}")
    for (line,) in cur.fetchall():
        print(f"   {line}")


def main():
    con = psycopg2.connect(**CFG)
    con.autocommit = True
    cur = con.cursor()

    print(f"== seed {SCALE} orders + 20k customers ==")
    cur.execute("DROP TABLE IF EXISTS lab05_orders")
    cur.execute("DROP TABLE IF EXISTS lab05_customers")
    cur.execute("CREATE TABLE lab05_customers (id INT PRIMARY KEY, region TEXT)")
    cur.execute("""CREATE TABLE lab05_orders (
        id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
        customer_id INT, total NUMERIC NOT NULL)  -- nullable on purpose: NULL-trap demo in §4""" )
    cur.execute("INSERT INTO lab05_customers SELECT g, "
                "CASE WHEN mod(g, 3) = 0 THEN 'south' ELSE 'north' END "
                "FROM generate_series(1, 20000) g")
    cur.execute(
        """INSERT INTO lab05_orders (customer_id, total)
           SELECT 1 + mod(g, 15000), mod(g, 500) + 1
           FROM generate_series(1, %s) g""", (SCALE,))
    cur.execute("CREATE INDEX lab05_o_cust ON lab05_orders(customer_id)")
    cur.execute("VACUUM ANALYZE lab05_orders")
    cur.execute("VACUUM ANALYZE lab05_customers")
    print("   seeded + indexed + ANALYZE")

    print("\n== 1. join algorithm follows outer size ==")
    explain(cur, "tiny outer (10 regions rows) -> Nested Loop + index probes",
            """SELECT o.id FROM lab05_orders o JOIN lab05_customers c ON c.id=o.customer_id
               WHERE c.id < 10 LIMIT 20""")
    explain(cur, "huge outer (all rows) -> Hash Join, one pass each side",
            """SELECT count(*) FROM lab05_orders o
               JOIN lab05_customers c ON c.id = o.customer_id""")

    print("\n== 2. work_mem cliff: same ORDER BY, tiny vs default work_mem ==")
    cur.execute("SET work_mem='64kB'")
    explain(cur, "work_mem=64kB -> external merge Disk + temp written (SLOW)",
            """SELECT id, customer_id, total FROM lab05_orders
               ORDER BY total LIMIT 5000""")
    cur.execute("RESET work_mem")
    explain(cur, "work_mem=4MB -> quicksort in memory (FAST)",
            """SELECT id, customer_id, total FROM lab05_orders
               ORDER BY total LIMIT 5000""")

    print("\n== 3. N+1: 200 single-row trips vs 1 join ==")
    cur.execute("SELECT id FROM lab05_customers LIMIT 200")
    ids = [r[0] for r in cur.fetchall()]
    t0 = time.perf_counter()
    n = 0
    for i in ids:
        cur.execute("SELECT count(*) FROM lab05_orders WHERE customer_id=%s", (i,))
        n += cur.fetchone()[0]
    dt_n1 = (time.perf_counter() - t0) * 1000
    t0 = time.perf_counter()
    cur.execute("SELECT count(*) FROM lab05_orders WHERE customer_id = ANY(%s)", (ids,))
    n2 = cur.fetchone()[0]
    dt_1 = (time.perf_counter() - t0) * 1000
    print(f"   N+1: 200 queries, total rows={n} in {dt_n1:.1f} ms")
    print(f"   1 query with = ANY: total rows={n2} in {dt_1:.1f} ms "
          f"(~{dt_n1/max(dt_1, 0.01):.0f}x faster)")

    print("\n== 4. EXISTS vs IN (+ NULL trap) ==")
    explain(cur, "EXISTS semi-join (stops at first match)",
            """SELECT count(*) FROM lab05_customers c WHERE EXISTS (
                 SELECT 1 FROM lab05_orders o WHERE o.customer_id = c.id)""")
    explain(cur, "IN (planner often rewrites to same semi-join — verify!)",
            """SELECT count(*) FROM lab05_customers c
               WHERE c.id IN (SELECT customer_id FROM lab05_orders)""")
    cur.execute("INSERT INTO lab05_orders (customer_id, total) VALUES (NULL, 1)")
    cur.execute("SELECT count(*) FROM lab05_customers WHERE id NOT IN "
                "(SELECT customer_id FROM lab05_orders)")
    print(f"   NOT IN with NULL present -> {cur.fetchone()[0]} rows (NULL poisoned it: expect 0!)")
    cur.execute("SELECT count(*) FROM lab05_customers c WHERE NOT EXISTS "
                "(SELECT 1 FROM lab05_orders o WHERE o.customer_id = c.id)")
    print(f"   NOT EXISTS (correct)      -> {cur.fetchone()[0]} rows")
    cur.execute("DELETE FROM lab05_orders WHERE customer_id IS NULL")

    print("\n== 5. CTE materialization control ==")
    explain(cur, "inlined CTE (default PG12+: filter pushes down)",
            """WITH big AS (SELECT customer_id, total FROM lab05_orders)
               SELECT count(*) FROM big WHERE customer_id < 100""")
    explain(cur, "MATERIALIZED CTE (computed fully, then filtered)",
            """WITH big AS MATERIALIZED (SELECT customer_id, total FROM lab05_orders)
               SELECT count(*) FROM big WHERE customer_id < 100""")

    print("\nDone. Takeaway: size picks the join; work_mem cliffs are visible in "
          "EXPLAIN; N+1 dies by round trips; NOT IN + NULL = 0 rows; CTE fence is opt-in.")
    cur.close()
    con.close()


if __name__ == "__main__":
    main()
