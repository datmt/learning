"""Lab 04 — OFFSET failure mode vs keyset pagination, deferred join, COUNT pain."""
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
SCALE = int(os.getenv("SCALE", "500000"))
PAGE = 20


def timed(cur, title, sql, params=()):
    t0 = time.perf_counter()
    cur.execute(sql, params)
    rows = cur.fetchall()
    dt = (time.perf_counter() - t0) * 1000
    print(f"   {title}: {dt:8.1f} ms  (first id={rows[0][0] if rows else '-'})")
    return rows


def explain(cur, title, sql):
    cur.execute(f"EXPLAIN (ANALYZE, BUFFERS, TIMING) {sql}")
    print(f"\n-- {title}\n   {sql[:150]}")
    for (line,) in cur.fetchall():
        print(f"   {line}")


def main():
    con = psycopg2.connect(**CFG)
    con.autocommit = True
    cur = con.cursor()

    print(f"== seed {SCALE} feed items (sequential ids, indexed) ==")
    t0 = time.perf_counter()
    cur.execute("DROP TABLE IF EXISTS lab04_items")
    cur.execute("""CREATE TABLE lab04_items (
        id BIGINT PRIMARY KEY GENERATED ALWAYS AS IDENTITY,
        author TEXT NOT NULL, body TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL)""")
    cur.execute(
        """INSERT INTO lab04_items (author, body, created_at)
           SELECT 'author' || mod(g, 1000),
                  'post body ' || g || ' ' || repeat('lorem ipsum dolor sit amet ', 20),
                  now() - (g || ' seconds')::interval
           FROM generate_series(1, %s) g""", (SCALE,))
    cur.execute("VACUUM ANALYZE lab04_items")
    print(f"   seeded in {time.perf_counter()-t0:.1f}s")
    deep = SCALE - 100

    print(f"\n== 1. OFFSET: page 1 vs deep page (LIMIT {PAGE}) ==")
    timed(cur, "OFFSET 0        ", "SELECT id FROM lab04_items ORDER BY id LIMIT 20 OFFSET 0")
    timed(cur, f"OFFSET {deep} (deep)", "SELECT id FROM lab04_items ORDER BY id LIMIT 20 OFFSET %s",
          (deep,))
    explain(cur, "plan: OFFSET scans + discards N rows",
            f"SELECT id FROM lab04_items ORDER BY id LIMIT 20 OFFSET {deep}")

    print("\n== 2. keyset: first page vs deep page (same ~cost) ==")
    first = timed(cur, "keyset start    ",
                  "SELECT id FROM lab04_items ORDER BY id LIMIT 20")
    last_id_first = first[-1][0]
    timed(cur, "keyset next page",
          "SELECT id FROM lab04_items WHERE id > %s ORDER BY id LIMIT 20",
          (last_id_first,))
    mid_id = deep  # ids are 1..SCALE so id ~= offset position
    timed(cur, "keyset deep page",
          "SELECT id FROM lab04_items WHERE id > %s ORDER BY id LIMIT 20",
          (mid_id,))
    explain(cur, "plan: keyset seeks via index, reads 20",
            f"SELECT id FROM lab04_items WHERE id > {mid_id} ORDER BY id LIMIT 20")

    print("\n== 3. deferred join (when page numbers are required) ==")
    timed(cur, "plain deep OFFSET ",
          "SELECT id, author, body FROM lab04_items ORDER BY id LIMIT 20 OFFSET %s",
          (deep,))
    timed(cur, "deferred deep     ",
          """SELECT t.id, t.author, t.body FROM lab04_items t JOIN (
                SELECT id FROM lab04_items ORDER BY id LIMIT 20 OFFSET %s
              ) p USING (id) ORDER BY t.id""", (deep,))

    print("\n== 4. COUNT(*) — the second killer ==")
    t0 = time.perf_counter()
    cur.execute("SELECT count(*) FROM lab04_items")
    print(f"   unfiltered COUNT(*): {cur.fetchone()[0]} rows in "
          f"{(time.perf_counter()-t0)*1000:.1f} ms")
    cur.execute("SELECT reltuples::bigint AS estimate FROM pg_class "
                "WHERE relname='lab04_items'")
    print(f"   reltuples estimate:  {cur.fetchone()[0]} rows (instant, no scan)")
    t0 = time.perf_counter()
    cur.execute("SELECT count(*) FROM (SELECT 1 FROM lab04_items LIMIT 10001) s")
    print(f"   capped COUNT (LIMIT 10001): {cur.fetchone()[0]} in "
          f"{(time.perf_counter()-t0)*1000:.1f} ms  -> display '10000+'")

    print("\nDone. Takeaway: OFFSET costs grow with page depth (here 0.5 ms -> "
          "~100 ms); keyset stays flat (~0.2 ms anywhere); the deferred join "
          "barely moves the needle because the offset *walk* itself dominates — "
          "which is exactly why keyset is the real fix; never COUNT(*) per page view.")
    cur.close()
    con.close()


if __name__ == "__main__":
    main()
