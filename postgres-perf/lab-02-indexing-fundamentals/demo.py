"""Lab 02 — B-tree wins, composite order, covering, ignored-index checklist, write cost."""
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
SCALE = int(os.getenv("SCALE", "200000"))


def explain(cur, title, sql):
    cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, TIMING) {sql}")
    print(f"\n-- {title}\n   {sql[:160]}")
    for (line,) in cur.fetchall():
        print(f"   {line}")


def main():
    con = psycopg2.connect(**CFG)
    con.autocommit = True
    cur = con.cursor()

    print(f"== seed {SCALE} rows, NO indexes yet ==")
    cur.execute("DROP TABLE IF EXISTS lab02_orders")
    cur.execute("""CREATE TABLE lab02_orders (
        id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
        customer_id INT NOT NULL, status TEXT NOT NULL,
        total NUMERIC NOT NULL, email TEXT NOT NULL)""")
    cur.execute(
        """INSERT INTO lab02_orders (customer_id, status, total, email)
           SELECT 1 + mod(g, 20000),
                  CASE WHEN mod(g, 100) = 0 THEN 'priority' ELSE 'standard' END,
                  mod(g, 500) + 1, 'user' || g || '@example.com'
           FROM generate_series(1, %s) g""", (SCALE,))
    cur.execute("VACUUM ANALYZE lab02_orders")
    print("   seeded + ANALYZE (deliberately no secondary indexes)")

    print("\n== 1. selective vs non-selective (same table, opposite plans) ==")
    explain(cur, "1% match WITHOUT index -> Seq Scan, Rows Removed ~99%",
            "SELECT count(*) FROM lab02_orders WHERE status='priority'")
    cur.execute("CREATE INDEX lab02_status ON lab02_orders(status)")
    cur.execute("VACUUM ANALYZE lab02_orders")
    explain(cur, "1% match WITH index -> Bitmap/Index Scan (~100x faster)",
            "SELECT count(*) FROM lab02_orders WHERE status='priority'")
    explain(cur, "99% match -> planner still Seq Scans (correct! random hops would lose)",
            "SELECT count(*) FROM lab02_orders WHERE status='standard'")

    print("\n== 2. composite leftmost prefix ==")
    cur.execute("CREATE INDEX lab02_cust_status ON lab02_orders(customer_id, status)")
    cur.execute("VACUUM ANALYZE lab02_orders")
    explain(cur, "WHERE customer_id=? uses (customer_id, status) [leftmost]",
            "SELECT count(*) FROM lab02_orders WHERE customer_id=42")
    explain(cur, "WHERE customer_id=? AND status=? uses same index fully",
            "SELECT count(*) FROM lab02_orders WHERE customer_id=42 AND status='priority'")
    explain(cur, "OR across columns -> Seq Scan (classic trap)",
            "SELECT count(*) FROM lab02_orders WHERE customer_id=42 OR status='priority'")
    explain(cur, "OR rewritten as UNION -> two index scans + Append",
            """SELECT count(*) FROM (
                 SELECT id FROM lab02_orders WHERE customer_id=42
                 UNION ALL
                 SELECT id FROM lab02_orders WHERE status='priority'
                   AND NOT (customer_id=42)) s""")

    print("\n== 3. covering index -> Index Only Scan ==")
    explain(cur, "before INCLUDE: Index Scan + heap fetches",
            "SELECT total FROM lab02_orders WHERE customer_id=42")
    cur.execute("CREATE INDEX lab02_cust_cover ON lab02_orders(customer_id) INCLUDE (total)")
    cur.execute("VACUUM ANALYZE lab02_orders")  # visibility map -> true index-only
    explain(cur, "after INCLUDE (total): Index Only Scan, Heap Fetches ~0",
            "SELECT total FROM lab02_orders WHERE customer_id=42")

    print("\n== 4. ignored-index checklist ==")
    cur.execute("CREATE INDEX lab02_email_pat ON lab02_orders(email text_pattern_ops)")
    cur.execute("VACUUM ANALYZE lab02_orders")
    explain(cur, "function on column kills B-tree use",
            "SELECT count(*) FROM lab02_orders WHERE lower(email)='user42@example.com'")
    explain(cur, "leading wildcard kills B-tree use",
            "SELECT count(*) FROM lab02_orders WHERE email LIKE '%example.com'")
    explain(cur, "trailing wildcard CAN use B-tree",
            "SELECT count(*) FROM lab02_orders WHERE email LIKE 'user42%'")

    print("\n== 5. price: write cost per index ==")
    for n_extra in (0, 3):
        cur.execute("DROP TABLE IF EXISTS lab02_w")
        cur.execute("CREATE TABLE lab02_w (id BIGINT GENERATED ALWAYS AS IDENTITY, a INT, b INT, c INT)")
        for i in range(n_extra):
            cur.execute(f"CREATE INDEX lab02_w_i{i} ON lab02_w(a, b, c)")
        t0 = time.perf_counter()
        cur.execute("INSERT INTO lab02_w (a,b,c) SELECT g, g+1, g+2 "
                    "FROM generate_series(1, 50000) g")
        dt = time.perf_counter() - t0
        cur.execute("SELECT pg_size_pretty(sum(pg_relation_size(indexrelid))::bigint) "
                    "FROM pg_index WHERE indrelid='lab02_w'::regclass")
        size = cur.fetchone()[0]
        print(f"   {n_extra} extra indexes: 50k inserts in {dt:.2f}s, index disk={size}")
    cur.execute("DROP TABLE IF EXISTS lab02_w")

    print("\nDone. Takeaway: index selective columns; composite serves leftmost; "
          "INCLUDE for hot paths; functions/leading-%/OR defeat B-trees; indexes tax writes.")
    cur.close()
    con.close()


if __name__ == "__main__":
    main()
