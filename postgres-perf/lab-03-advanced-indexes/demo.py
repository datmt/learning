"""Lab 03 — partial, expression, GIN (jsonb/full-text/trgm), BRIN, bloat."""
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


def explain(cur, title, sql):
    cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, TIMING) {sql}")
    print(f"\n-- {title}\n   {sql[:170]}")
    for (line,) in cur.fetchall():
        print(f"   {line}")


def size(cur, idx):
    cur.execute("SELECT pg_size_pretty(pg_relation_size(%s))", (idx,))
    return cur.fetchone()[0]


def main():
    con = psycopg2.connect(**CFG)
    con.autocommit = True
    cur = con.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS pg_trgm")

    print(f"== seed {SCALE} events (2% open, jsonb payload, text body) ==")
    cur.execute("DROP TABLE IF EXISTS lab03_events")
    cur.execute("""CREATE TABLE lab03_events (
        id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
        status TEXT NOT NULL, customer_id INT NOT NULL,
        email TEXT NOT NULL, payload JSONB NOT NULL,
        body TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL)""")
    cur.execute(
        """INSERT INTO lab03_events (status, customer_id, email, payload, body, created_at)
           SELECT CASE WHEN mod(g, 50) = 0 THEN 'open' ELSE 'closed' END,
                  1 + mod(g, 20000),
                  'User' || g || '@Example.COM',
                  jsonb_build_object('vip', mod(g, 500) = 0, 'tier', mod(g, 5)),
                  CASE WHEN mod(g, 500) = 0
                       THEN 'customer requests refund after long shipping delay complaint'
                       ELSE 'routine order update notification message' END,
                  now() - (g || ' seconds')::interval
           FROM generate_series(1, %s) g""", (SCALE,))
    cur.execute("VACUUM ANALYZE lab03_events")
    print("   seeded + ANALYZE")

    print("\n== 1. partial index (hot 2% subset) ==")
    explain(cur, "before: hot-subset query Seq Scans 200k",
            "SELECT count(*) FROM lab03_events WHERE status='open' AND customer_id=42")
    cur.execute("CREATE INDEX lab03_partial ON lab03_events(customer_id) WHERE status='open'")
    cur.execute("VACUUM ANALYZE lab03_events")
    explain(cur, "after: Bitmap/Index on tiny partial index",
            "SELECT count(*) FROM lab03_events WHERE status='open' AND customer_id=42")
    explain(cur, "cold-subset query correctly ignores it (full scan or other plan)",
            "SELECT count(*) FROM lab03_events WHERE status='closed' AND customer_id=42")
    cur.execute("CREATE INDEX lab03_full_tmp ON lab03_events(customer_id)")
    print(f"   size partial={size(cur,'lab03_partial')} vs full-column={size(cur,'lab03_full_tmp')}")
    cur.execute("DROP INDEX lab03_full_tmp")

    print("\n== 2. expression index (lower(email)) ==")
    explain(cur, "before: lower() Seq Scans",
            "SELECT count(*) FROM lab03_events WHERE lower(email)='user42@example.com'")
    cur.execute("CREATE INDEX lab03_lower_email ON lab03_events(lower(email))")
    cur.execute("VACUUM ANALYZE lab03_events")
    explain(cur, "after: Index Scan on expression",
            "SELECT count(*) FROM lab03_events WHERE lower(email)='user42@example.com'")

    print("\n== 3. GIN: jsonb + full-text + substring ==")
    explain(cur, "jsonb @> before GIN (Seq Scan)",
            "SELECT count(*) FROM lab03_events WHERE payload @> '{\"vip\": true}'")
    cur.execute("CREATE INDEX lab03_gin_payload ON lab03_events USING gin (payload)")
    cur.execute("VACUUM ANALYZE lab03_events")
    explain(cur, "jsonb @> after GIN (Bitmap)",
            "SELECT count(*) FROM lab03_events WHERE payload @> '{\"vip\": true}'")
    explain(cur, "full-text @@ before GIN (Seq Scan + tsvector filter)",
            "SELECT count(*) FROM lab03_events WHERE "
            "to_tsvector('english', body) @@ to_tsquery('english', 'refund & delay')")
    cur.execute("CREATE INDEX lab03_gin_fts ON lab03_events "
                "USING gin (to_tsvector('english', body))")
    cur.execute("VACUUM ANALYZE lab03_events")
    explain(cur, "full-text @@ after GIN (Bitmap)",
            "SELECT count(*) FROM lab03_events WHERE "
            "to_tsvector('english', body) @@ to_tsquery('english', 'refund & delay')")
    explain(cur, "substring LIKE '%...%' before trgm (Seq Scan)",
            "SELECT count(*) FROM lab03_events WHERE email LIKE '%ample.C%'")
    cur.execute("CREATE INDEX lab03_gin_trgm ON lab03_events USING gin (email gin_trgm_ops)")
    cur.execute("VACUUM ANALYZE lab03_events")
    explain(cur, "substring after trgm GIN (Bitmap)",
            "SELECT count(*) FROM lab03_events WHERE email LIKE '%ample.C%'")

    print("\n== 4. BRIN for append-only time series ==")
    cur.execute("DROP TABLE IF EXISTS lab03_metrics")
    cur.execute("""CREATE TABLE lab03_metrics (
        ts TIMESTAMPTZ NOT NULL, value DOUBLE PRECISION NOT NULL)""")
    cur.execute("""INSERT INTO lab03_metrics
                   SELECT now() - (g || ' seconds')::interval, mod(g, 100) + 0.5
                   FROM generate_series(1, %s) g""", (SCALE,))
    cur.execute("VACUUM ANALYZE lab03_metrics")
    cur.execute("CREATE INDEX lab03_brin ON lab03_metrics USING brin (ts)")
    cur.execute("CREATE INDEX lab03_btree_ts ON lab03_metrics (ts)")
    cur.execute("VACUUM ANALYZE lab03_metrics")
    print(f"   size BRIN={size(cur,'lab03_brin')} vs B-tree={size(cur,'lab03_btree_ts')}")
    cur.execute("DROP INDEX lab03_btree_ts")
    explain(cur, "1-hour range over 200k uses tiny BRIN (block skipping)",
            "SELECT count(*), avg(value) FROM lab03_metrics "
            "WHERE ts > now() - interval '1 hour'")

    print("\n== 5. bloat: churn, then REINDEX ==")
    print(f"   partial index before churn: {size(cur,'lab03_partial')}")
    cur.execute("UPDATE lab03_events SET status = CASE WHEN status='open' THEN 'closed' "
                "ELSE 'open' END WHERE mod(id, 7) = 0")
    cur.execute("VACUUM lab03_events")  # reclaims heap, index pages stay half-empty
    print(f"   after churn + VACUUM:       {size(cur,'lab03_partial')} (often larger)")
    cur.execute("REINDEX INDEX lab03_partial")
    print(f"   after REINDEX:              {size(cur,'lab03_partial')} (compact again)")

    print("\nDone. Takeaway: partial=hot subset, expression=computed value, "
          "GIN=inside-the-value search, BRIN=ordered time series, REINDEX fixes bloat.")
    cur.close()
    con.close()


if __name__ == "__main__":
    main()
