"""Lab 7 — Modern SQL features: JSON, FTS5, window functions, CTEs, UPSERT, RETURNING, generated columns."""
import sqlite3

db = sqlite3.connect(":memory:", autocommit=True)


def show(title, sql, params=()):
    print(f"\n-- {title}\n   {sql.strip()}")
    try:
        for row in db.execute(sql, params):
            print("  ", row)
    except sqlite3.Error as e:
        print("   ERROR:", type(e).__name__, e)


print("SQLite", sqlite3.sqlite_version)

print("\n== 1. JSON")
db.execute("CREATE TABLE ev (id INTEGER PRIMARY KEY, body TEXT)")
db.executemany("INSERT INTO ev (body) VALUES (?)", [
    ('{"user": {"name": "ann", "tier": "pro"}, "tags": ["a", "b"], "ms": 120}',),
    ('{"user": {"name": "bob", "tier": "free"}, "tags": ["b"], "ms": 340}',),
    ('{"user": {"name": "cid", "tier": "pro"}, "tags": [], "ms": 80}',),
])
show("extract with ->> (SQL value) and -> (JSON text)",
     "SELECT body ->> '$.user.name', body -> '$.tags', body ->> '$.ms' + 0 FROM ev")
show("filter + aggregate on JSON", "SELECT body ->> '$.user.tier' AS tier, avg(body ->> '$.ms') FROM ev GROUP BY tier")
show("json_each: unnest an array", "SELECT ev.id, j.value FROM ev, json_each(ev.body, '$.tags') AS j")
show("build JSON", "SELECT json_group_array(json_object('id', id, 'name', body ->> '$.user.name')) FROM ev")
show("modify JSON", "SELECT json_set(body, '$.ms', 0, '$.new', 'x') FROM ev WHERE id = 1")
show("invalid JSON check", "SELECT json_valid('{bad'), json_valid('{\"ok\":1}')")
db.execute("CREATE INDEX ix_tier ON ev (body ->> '$.user.tier')")
show("index on a JSON path",
     "EXPLAIN QUERY PLAN SELECT id FROM ev WHERE body ->> '$.user.tier' = 'pro'")
db.execute("ALTER TABLE ev ADD COLUMN tier TEXT GENERATED ALWAYS AS (body ->> '$.user.tier') VIRTUAL")
show("generated column from JSON", "SELECT id, tier FROM ev")
show("JSONB (3.45+): binary JSON, faster to re-parse", "SELECT typeof(jsonb('{\"a\":1}')), json(jsonb('{\"a\":1}'))")

print("\n== 2. FTS5 full-text search")
db.execute("CREATE VIRTUAL TABLE docs USING fts5(title, body, tokenize='porter unicode61')")
db.executemany("INSERT INTO docs VALUES (?, ?)", [
    ("SQLite WAL", "Write-ahead logging lets readers run while a writer commits."),
    ("Indexes", "A covering index answers queries without touching the table."),
    ("Backups", "Use the online backup API or VACUUM INTO while the database is running."),
    ("Locking", "Only one writer at a time. Readers never block in WAL mode."),
])
show("MATCH with stemming ('writers' finds 'writer')",
     "SELECT title FROM docs WHERE docs MATCH 'writers'")
show("boolean + phrase + prefix",
     "SELECT title FROM docs WHERE docs MATCH '(\"backup API\" OR index*) NOT locking'")
show("ranked (bm25) with highlighted snippet",
     "SELECT title, snippet(docs, 1, '[', ']', '...', 6) FROM docs WHERE docs MATCH 'reader*' ORDER BY rank")
show("column filter", "SELECT title FROM docs WHERE docs MATCH 'title: wal'")

print("\n== 3. Window functions")
db.execute("CREATE TABLE sales (day INT, region TEXT, amount INT)")
db.executemany("INSERT INTO sales VALUES (?,?,?)",
               [(d, r, a) for d, r, a in [(1, 'eu', 10), (2, 'eu', 30), (3, 'eu', 20),
                                          (1, 'us', 50), (2, 'us', 5), (3, 'us', 40)]])
show("running total, rank, previous value", """
SELECT region, day, amount,
       sum(amount) OVER (PARTITION BY region ORDER BY day)         AS running,
       rank()      OVER (PARTITION BY region ORDER BY amount DESC) AS rnk,
       lag(amount) OVER (PARTITION BY region ORDER BY day)         AS prev
FROM sales ORDER BY region, day""")

print("\n== 4. Recursive CTE")
show("generate a series", "WITH RECURSIVE n(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM n WHERE x < 5) SELECT group_concat(x) FROM n")
db.execute("CREATE TABLE emp (id INT, name TEXT, boss INT)")
db.executemany("INSERT INTO emp VALUES (?,?,?)", [(1, 'ceo', None), (2, 'cto', 1), (3, 'dev', 2), (4, 'intern', 3)])
show("walk a tree", """
WITH RECURSIVE chain(id, name, depth) AS (
  SELECT id, name, 0 FROM emp WHERE boss IS NULL
  UNION ALL SELECT e.id, e.name, depth + 1 FROM emp e JOIN chain c ON e.boss = c.id)
SELECT substr('        ', 1, depth * 2) || name FROM chain""")

print("\n== 5. UPSERT and RETURNING")
db.execute("CREATE TABLE counter (key TEXT PRIMARY KEY, hits INT)")
for _ in range(3):
    db.execute("INSERT INTO counter VALUES ('home', 1) ON CONFLICT(key) DO UPDATE SET hits = hits + 1")
show("upsert x3", "SELECT * FROM counter")
show("RETURNING gives back generated values", "INSERT INTO ev (body) VALUES ('{}') RETURNING id, tier")
show("DELETE ... RETURNING", "DELETE FROM counter RETURNING *")

print("\n== 6. Misc")
show("math functions", "SELECT round(sqrt(2), 4), pow(2, 10), ln(exp(1))")
show("string_agg / iif / format", "SELECT string_agg(region, ',') , iif(1 > 0, 'yes', 'no'), format('%05d', 42) FROM (SELECT DISTINCT region FROM sales)")
show("FILTER clause on aggregates",
     "SELECT count(*) FILTER (WHERE amount > 20), count(*) FROM sales")
show("RIGHT/FULL OUTER JOIN (3.39+)",
     "SELECT a.x, b.x FROM (SELECT 1 x UNION SELECT 2) a FULL JOIN (SELECT 2 x UNION SELECT 3) b ON a.x = b.x")
