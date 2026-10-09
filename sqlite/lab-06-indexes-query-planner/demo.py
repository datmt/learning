"""Lab 6 — Indexes and the query planner: EXPLAIN QUERY PLAN, covering, composite, partial, expression."""
import random
import sqlite3
import time

db = sqlite3.connect(":memory:", autocommit=True)
random.seed(1)
db.execute("""CREATE TABLE orders (
    id INTEGER PRIMARY KEY, customer_id INT, status TEXT, total REAL, created_at INT, email TEXT)""")
statuses = ["done"] * 97 + ["pending"] * 2 + ["failed"]
db.execute("BEGIN")
db.executemany("INSERT INTO orders VALUES (?,?,?,?,?,?)", (
    (i, random.randint(1, 50_000), random.choice(statuses), round(random.random() * 500, 2),
     1_700_000_000 + i, f"User{i}@Mail.com") for i in range(1, 500_001)))
db.execute("COMMIT")


def plan(sql, params=()):
    return " | ".join(r[3] for r in db.execute("EXPLAIN QUERY PLAN " + sql, params))


def check(title, sql, params=()):
    t0 = time.perf_counter()
    for _ in range(20):
        db.execute(sql, params).fetchall()
    ms = (time.perf_counter() - t0) / 20 * 1000
    print(f"\n-- {title}\n   {sql}\n   plan: {plan(sql, params)}\n   time: {ms:.3f} ms")


q = "SELECT * FROM orders WHERE customer_id = ?"
check("1. No index -> full table SCAN", q, (42,))
db.execute("CREATE INDEX ix_cust ON orders(customer_id)")
check("   With index -> SEARCH", q, (42,))
check("   rowid lookup is the fastest path", "SELECT * FROM orders WHERE id = ?", (42,))

print("\n== 2. Composite index: column order matters (leftmost prefix rule)")
db.execute("CREATE INDEX ix_status_created ON orders(status, created_at)")
check("uses both columns", "SELECT id FROM orders WHERE status = 'pending' AND created_at > ?", (1_700_400_000,))
check("only 2nd column -> index useless (scan)", "SELECT id FROM orders WHERE created_at > ?", (1_700_499_990,))

print("\n== 3. Covering index: answer straight from the index, skip the table")
check("needs table lookup per row", "SELECT customer_id, total FROM orders WHERE customer_id BETWEEN 100 AND 2000")
db.execute("CREATE INDEX ix_cust_total ON orders(customer_id, total)")
check("COVERING INDEX", "SELECT customer_id, total FROM orders WHERE customer_id BETWEEN 100 AND 2000")

print("\n== 4. Functions on a column defeat the index")
db.execute("CREATE INDEX ix_email ON orders(email)")
check("lower(email) -> scan", "SELECT id FROM orders WHERE lower(email) = ?", ("user42@mail.com",))
db.execute("CREATE INDEX ix_email_lower ON orders(lower(email))")
check("expression index on lower(email) -> search", "SELECT id FROM orders WHERE lower(email) = ?", ("user42@mail.com",))

print("\n== 5. Type mismatch: TEXT column compared to integer")
db.execute("CREATE TABLE codes (code TEXT)")
db.executemany("INSERT INTO codes VALUES (?)", ((str(i),) for i in range(100_000)))
db.execute("CREATE INDEX ix_code ON codes(code)")
check("TEXT affinity converts 123 -> '123', index still used", "SELECT * FROM codes WHERE code = 123")
check("but CAST on the column kills it", "SELECT * FROM codes WHERE CAST(code AS INT) = 123")

print("\n== 6. Partial index: index only the rows you query")
db.execute("DROP INDEX ix_status_created")  # otherwise the planner prefers it here
db.execute("CREATE INDEX ix_pending ON orders(created_at) WHERE status = 'pending'")
db.execute("CREATE INDEX ix_all_created ON orders(created_at)")
check("partial index used (query WHERE implies index WHERE)",
      "SELECT id FROM orders WHERE status = 'pending' ORDER BY created_at")
check("not usable when the query doesn't imply status='pending'",
      "SELECT id FROM orders WHERE status = 'failed' ORDER BY created_at")
print("   index sizes (pages):", db.execute(
    "SELECT name, count(*) FROM dbstat WHERE name IN ('ix_pending','ix_all_created') GROUP BY name").fetchall())
db.execute("DROP INDEX ix_all_created")

print("\n== 7. OR / LIKE / leading wildcard")
check("OR on two indexed columns -> MULTI-INDEX OR", "SELECT id FROM orders WHERE customer_id = 7 OR email = 'User9@Mail.com'")
check("LIKE 'x%' can't use a normal index (LIKE is case-insensitive)", "SELECT id FROM orders WHERE email LIKE 'User42@%'")
check("GLOB prefix (case-sensitive) can use it", "SELECT id FROM orders WHERE email GLOB 'User42@*'")
check("leading wildcard never can", "SELECT id FROM orders WHERE email LIKE '%42@Mail.com'")

print("\n== 8. ORDER BY + LIMIT: index avoids sorting (no 'USE TEMP B-TREE')")
check("sort needed", "SELECT id FROM orders ORDER BY total DESC LIMIT 10")
db.execute("CREATE INDEX ix_total ON orders(total)")
check("index provides order", "SELECT id FROM orders ORDER BY total DESC LIMIT 10")

print("\n== 9. ANALYZE gives the planner statistics (sqlite_stat1)")
db.execute("CREATE INDEX ix_status ON orders(status)")   # low-selectivity index: 3 distinct values
q = "SELECT id FROM orders WHERE status = 'done' AND created_at > ?"
check("before ANALYZE: planner guesses", q, (1_700_499_000,))
db.execute("ANALYZE")
print("  ", db.execute("SELECT idx, stat FROM sqlite_stat1 WHERE idx IN ('ix_status','ix_cust')").fetchall())
print("   stat = 'total_rows  avg_rows_per_key'. ix_status: 166,667 rows per value -> nearly useless")
check("after ANALYZE", q, (1_700_499_000,))
