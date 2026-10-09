"""Lab 8 — Hard limits: test each one until it breaks."""
import os
import sqlite3
import tempfile

db = sqlite3.connect(":memory:", autocommit=True)


def try_it(title, fn):
    try:
        print(f"   {title}: OK -> {fn()}")
    except (sqlite3.Error, OverflowError, MemoryError) as e:
        print(f"   {title}: {type(e).__name__}: {e}")


print("== 1. Compile-time limits of this build (sqlite3.connection.getlimit)")
for name in ["LENGTH", "SQL_LENGTH", "COLUMN", "EXPR_DEPTH", "COMPOUND_SELECT",
             "VARIABLE_NUMBER", "ATTACHED", "LIKE_PATTERN_LENGTH", "TRIGGER_DEPTH"]:
    print(f"   SQLITE_LIMIT_{name:20} = {db.getlimit(getattr(sqlite3, 'SQLITE_LIMIT_' + name)):,}")

print("\n== 2. Max columns per table (SQLITE_LIMIT_COLUMN)")
n = db.getlimit(sqlite3.SQLITE_LIMIT_COLUMN)
try_it(f"{n} columns", lambda: db.execute(
    f"CREATE TABLE wide ({', '.join(f'c{i}' for i in range(n))})") and "created")
try_it(f"{n + 1} columns", lambda: db.execute(
    f"CREATE TABLE wider ({', '.join(f'c{i}' for i in range(n + 1))})"))

print("\n== 3. Max bound parameters (SQLITE_LIMIT_VARIABLE_NUMBER)")
n = db.getlimit(sqlite3.SQLITE_LIMIT_VARIABLE_NUMBER)
db.execute("CREATE TABLE ids (id INTEGER)")
try_it(f"IN with {n} params", lambda: db.execute(
    f"SELECT count(*) FROM ids WHERE id IN ({','.join('?' * n)})", range(n)).fetchone())
try_it(f"IN with {n + 1} params", lambda: db.execute(
    f"SELECT count(*) FROM ids WHERE id IN ({','.join('?' * (n + 1))})", range(n + 1)).fetchone())
print("   fix for huge lists: json_each or a temp table:")
try_it("100k ids via json_each", lambda: db.execute(
    "SELECT count(*) FROM json_each(?)", (str(list(range(100_000))),)).fetchone())

print("\n== 4. Max string/blob size (SQLITE_LIMIT_LENGTH). Lowered to 1 MB so the test is cheap")
db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000)
try_it("999,000 byte blob", lambda: db.execute("SELECT length(zeroblob(999000))").fetchone())
try_it("1,000,001 byte blob", lambda: db.execute("SELECT length(zeroblob(1000001))").fetchone())
db.setlimit(sqlite3.SQLITE_LIMIT_LENGTH, 1_000_000_000)

print("\n== 5. Expression depth (SQLITE_LIMIT_EXPR_DEPTH)")
n = db.getlimit(sqlite3.SQLITE_LIMIT_EXPR_DEPTH)
try_it(f"{n - 2} nested additions", lambda: db.execute("SELECT " + "1+" * (n - 2) + "1").fetchone())
try_it(f"{n + 1} nested additions", lambda: db.execute("SELECT " + "1+" * (n + 1) + "1").fetchone())

print("\n== 6. Compound SELECT terms (SQLITE_LIMIT_COMPOUND_SELECT): hits multi-row INSERT ... UNION ALL")
n = db.getlimit(sqlite3.SQLITE_LIMIT_COMPOUND_SELECT)
try_it(f"{n} UNION ALL terms", lambda: db.execute(" UNION ALL ".join(["SELECT 1"] * n)).fetchall().__len__())
try_it(f"{n + 1} UNION ALL terms", lambda: db.execute(" UNION ALL ".join(["SELECT 1"] * (n + 1))).fetchall())
try_it("VALUES with 100k rows is fine (not a compound)", lambda: db.execute(
    "SELECT count(*) FROM (VALUES " + ",".join(["(1)"] * 100_000) + ")").fetchone())

print("\n== 7. ATTACH limit (default 10)")
d = tempfile.mkdtemp()
n = db.getlimit(sqlite3.SQLITE_LIMIT_ATTACHED)
for i in range(n):
    db.execute(f"ATTACH ? AS a{i}", (os.path.join(d, f"a{i}.db"),))
try_it(f"attach #{n + 1}", lambda: db.execute("ATTACH ? AS extra", (os.path.join(d, "x.db"),)))

print("\n== 8. Integer range: 64-bit signed")
try_it("max int64", lambda: db.execute("SELECT 9223372036854775807, typeof(9223372036854775807)").fetchone())
try_it("one more literal -> REAL", lambda: db.execute("SELECT typeof(9223372036854775808)").fetchone())
try_it("Python int too big to bind", lambda: db.execute("SELECT ?", (2 ** 63,)).fetchone())

print("\n== 9. Database size = max_page_count x page_size")
f = sqlite3.connect(os.path.join(d, "size.db"), autocommit=True)
ps, mpc = f.execute("PRAGMA page_size").fetchone()[0], f.execute("PRAGMA max_page_count").fetchone()[0]
print(f"   page_size={ps}, max_page_count={mpc:,} -> max {ps * mpc / 2**40:.1f} TiB "
      f"(65536-byte pages -> {65536 * mpc / 2**40:.0f} TiB)")
f.execute("CREATE TABLE t (b BLOB)")
f.execute("PRAGMA max_page_count = 100")  # cap at 400 KB to simulate a full disk
try_it("insert 1 MB into a db capped at 100 pages", lambda: f.execute("INSERT INTO t VALUES (zeroblob(1000000))"))

print("\n== 10. rowid max is 2^63-1; once used, SQLite picks random unused ones")
db.execute("CREATE TABLE r (id INTEGER PRIMARY KEY)")
db.execute("INSERT INTO r VALUES (9223372036854775807)")
try_it("insert after max rowid", lambda: db.execute("INSERT INTO r VALUES (NULL) RETURNING id").fetchone())
db.execute("CREATE TABLE ra (id INTEGER PRIMARY KEY AUTOINCREMENT)")
db.execute("INSERT INTO ra VALUES (9223372036854775807)")
try_it("same with AUTOINCREMENT", lambda: db.execute("INSERT INTO ra VALUES (NULL)"))
