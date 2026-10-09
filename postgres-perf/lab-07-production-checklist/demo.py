"""Lab 07 — audit playbook: pg_stat_statements, unused/missing indexes, MV, partitioning."""
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


def show(cur, title, sql, params=()):
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    print(f"\n-- {title}\n   {sql.strip()[:160]}")
    print("   | ".join(cols))
    for r in rows[:12]:
        print("   | ".join("NULL" if v is None else str(v)[:55] for v in r))


def timed(cur, title, sql):
    t0 = time.perf_counter()
    cur.execute(sql)
    n = cur.fetchall()
    print(f"   {title}: {(time.perf_counter()-t0)*1000:.1f} ms -> {str(n[0][0])[:40]}")


def main():
    con = psycopg2.connect(**CFG)
    con.autocommit = True
    cur = con.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")

    print(f"== seed events ({SCALE}) + create a deliberate mess ==")
    cur.execute("DROP MATERIALIZED VIEW IF EXISTS lab07_daily_revenue")
    cur.execute("DROP TABLE IF EXISTS lab07_events")
    cur.execute("""CREATE TABLE lab07_events (
        id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
        customer_id INT NOT NULL, status TEXT NOT NULL,
        total NUMERIC NOT NULL, created_at TIMESTAMPTZ NOT NULL)""")
    cur.execute(
        """INSERT INTO lab07_events (customer_id, status, total, created_at)
           SELECT 1 + mod(g, 20000),
                  CASE WHEN mod(g, 50) = 0 THEN 'open' ELSE 'closed' END,
                  mod(g, 500) + 1, now() - (mod(g, 90) || ' days')::interval
           FROM generate_series(1, %s) g""", (SCALE,))
    cur.execute("CREATE INDEX lab07_status ON lab07_events(status)")
    cur.execute("CREATE INDEX lab07_never_used ON lab07_events(total)")  # bait for §2
    cur.execute("VACUUM ANALYZE lab07_events")
    print("   seeded + ANALYZE (+ one index nothing will ever use)")

    print("\n== warm up pg_stat_statements with good + bad queries ==")
    for _ in range(5):
        cur.execute("SELECT count(*) FROM lab07_events WHERE status='open'")       # fast, indexed
    cur.execute("""SELECT customer_id, sum(total) FROM lab07_events
                   GROUP BY customer_id ORDER BY sum(total) DESC LIMIT 50""")      # heavy dashboard
    cur.execute("SELECT count(*) FROM lab07_events WHERE total::text LIKE '%%99%%'")  # unindexable
    # pg_stat_* counters flush async — force it so §2/§3 read fresh numbers
    cur.execute("SELECT pg_stat_force_next_flush()")

    print("\n== 1. where is time going? (sort by TOTAL, not mean) ==")
    show(cur, "top statements by total time",
         """SELECT left(query, 55) AS query, calls,
                   round(total_exec_time::numeric) AS total_ms,
                   round(mean_exec_time::numeric, 2) AS mean_ms
            FROM pg_stat_statements
            WHERE dbid = (SELECT oid FROM pg_database WHERE datname=current_database())
              AND query LIKE '%%lab07%%'
            ORDER BY total_exec_time DESC LIMIT 8""")

    print("\n== 2. unused indexes (pure write tax) ==")
    show(cur, "indexes never scanned (drop candidates)",
         """SELECT indexrelname AS index_,
                   pg_size_pretty(pg_relation_size(indexrelid)) AS size
            FROM pg_stat_user_indexes
            WHERE relname='lab07_events' AND idx_scan=0
              AND indexrelname NOT LIKE '%%pkey%%'""")

    print("\n== 3. missing indexes + bloat signals ==")
    show(cur, "seq scans / rows / dead tuples per table",
         """SELECT relname AS table_, seq_scan, seq_tup_read, n_dead_tup,
                   pg_size_pretty(pg_total_relation_size(relid)) AS total_size
            FROM pg_stat_user_tables
            WHERE relname LIKE 'lab07%%' ORDER BY seq_tup_read DESC""")

    print("\n== 4. materialized view: pre-compute the dashboard ==")
    cur.execute("DROP MATERIALIZED VIEW IF EXISTS lab07_daily_revenue")
    cur.execute("""CREATE MATERIALIZED VIEW lab07_daily_revenue AS
                   SELECT created_at::date AS day, count(*) AS orders, sum(total) AS revenue
                   FROM lab07_events GROUP BY 1""")
    cur.execute("CREATE UNIQUE INDEX ON lab07_daily_revenue(day)")
    timed(cur, "raw GROUP BY over 200k ",
          "SELECT count(*), sum(revenue) FROM (SELECT created_at::date d, sum(total) revenue "
          "FROM lab07_events GROUP BY 1) s")
    timed(cur, "SELECT from MV           ",
          "SELECT count(*), sum(revenue) FROM lab07_daily_revenue")
    t0 = time.perf_counter()
    cur.execute("REFRESH MATERIALIZED VIEW CONCURRENTLY lab07_daily_revenue")
    print(f"   CONCURRENTLY refresh: {(time.perf_counter()-t0)*1000:.1f} ms (no read lock)")

    print("\n== 5. partitioning: prune 2 of 3 partitions ==")
    cur.execute("DROP TABLE IF EXISTS lab07_sales")
    cur.execute("""CREATE TABLE lab07_sales (id BIGINT, month DATE, amount NUMERIC)
                   PARTITION BY RANGE (month)""")
    for m in ("2026-01-01", "2026-02-01", "2026-03-01"):
        cur.execute(f"CREATE TABLE lab07_sales_{m[:7].replace('-', '_')} "
                    f"PARTITION OF lab07_sales FOR VALUES FROM ('{m}') TO "
                    f"('{m}'::date + interval '1 month')")
    cur.execute("""INSERT INTO lab07_sales
                   SELECT g, date '2026-01-01' + (mod(g, 90) || ' days')::interval, mod(g, 500)+1
                   FROM generate_series(1, 90000) g""")
    cur.execute("VACUUM ANALYZE lab07_sales")
    cur.execute("EXPLAIN (ANALYZE, BUFFERS, TIMING) SELECT sum(amount) FROM lab07_sales "
                "WHERE month >= date '2026-02-01' AND month < date '2026-03-01'")
    print("   -- one-month query (expect single partition lab07_sales_2026_02):")
    for (line,) in cur.fetchall():
        print(f"   {line}")

    print("\nDone. Takeaway: rank by total time; drop idx_scan=0; watch seq_tup_read + "
          "n_dead_tup; MV the dashboard; partition by time. Steal these queries for runbooks.")
    cur.close()
    con.close()


if __name__ == "__main__":
    main()
