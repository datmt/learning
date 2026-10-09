"""Lab 00 — infra check + measurement habit. Verifies PG16, settings, seeds data."""
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
SCALE = int(os.getenv("SCALE", "50000"))


def q(cur, sql, params=()):
    cur.execute(sql, params)
    try:
        return cur.fetchall()
    except psycopg2.ProgrammingError:
        return []  # e.g. EXPLAIN without RETURNING rows handled separately


def show(cur, title, sql, params=()):
    cur.execute(sql, params)
    cols = [d[0] for d in cur.description]
    rows = cur.fetchall()
    print(f"\n-- {title}\n   {sql.strip()[:200]}")
    print("   | ".join(cols))
    for r in rows[:10]:
        print("   | ".join("NULL" if v is None else str(v)[:60] for v in r))
    return rows


def explain(cur, title, sql):
    cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, TIMING) {sql}")
    print(f"\n-- {title}")
    for (line,) in cur.fetchall():
        print(f"   {line}")


def main():
    con = psycopg2.connect(**CFG)
    con.autocommit = True
    cur = con.cursor()

    print("== 1. server + extension check ==")
    show(cur, "version (expect PostgreSQL 16.x)", "SELECT version()")
    show(cur, "pg_stat_statements present?",
         "SELECT count(*) AS ext FROM pg_extension WHERE extname='pg_stat_statements'")
    # extension may exist but not created in this DB — create it (compose preloads the lib)
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_stat_statements")

    print("\n== 2. settings that shape every benchmark ==")
    show(cur, "memory/planner settings",
         "SELECT name, setting, unit FROM pg_settings "
         "WHERE name IN ('shared_buffers','work_mem','random_page_cost',"
         "'effective_cache_size','log_min_duration_statement') ORDER BY name")

    print(f"\n== 3. seed {SCALE} rows (server-side generate_series, fast) ==")
    t0 = time.perf_counter()
    cur.execute("DROP TABLE IF EXISTS lab00_events")
    cur.execute("""CREATE TABLE lab00_events (
        id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
        status TEXT NOT NULL, amount NUMERIC NOT NULL, created_at TIMESTAMPTZ NOT NULL)""")
    # 2% 'rare', 98% 'common' — skewed on purpose (labs 01-02 reuse this trick)
    cur.execute(
        """INSERT INTO lab00_events (status, amount, created_at)
           SELECT CASE WHEN mod(g, 50) = 0 THEN 'rare' ELSE 'common' END,
                  mod(g, 1000) + 0.5,
                  now() - (g || ' minutes')::interval
           FROM generate_series(1, %s) g""", (SCALE,))
    cur.execute("VACUUM ANALYZE lab00_events")
    print(f"   seeded {SCALE} rows in {time.perf_counter()-t0:.2f}s + VACUUM ANALYZE")

    print("\n== 4. three ways to time the same query ==")
    sql = "SELECT count(*) FROM lab00_events WHERE status='rare'"
    t0 = time.perf_counter()
    cur.execute(sql)
    n = cur.fetchone()[0]
    print(f"   a) client timing: count={n} in {(time.perf_counter()-t0)*1000:.1f} ms "
          "(includes network; run-to-run noise)")
    explain(cur, "b) EXPLAIN ANALYZE (server time + plan, cold — expect shared read)", sql)
    explain(cur, "c) same query warm (expect shared hit, faster)", sql)

    print("\n== 5. pg_stat_statements sees us ==")
    show(cur, "top statements by calls (ours should appear)",
         """SELECT left(query, 60) AS query, calls, round(total_exec_time::numeric,1) AS total_ms
            FROM pg_stat_statements WHERE dbid = (SELECT oid FROM pg_database WHERE datname=current_database())
            ORDER BY total_exec_time DESC LIMIT 5""")

    print("\nDone. Habit: EXPLAIN (ANALYZE, BUFFERS) twice (cold+warm) for every "
          "claim from now on. Next: lab-01-query-plan-reading.")
    cur.close()
    con.close()


if __name__ == "__main__":
    main()
