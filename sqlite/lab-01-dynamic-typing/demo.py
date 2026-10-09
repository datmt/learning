"""Lab 1 — Dynamic typing, type affinity, STRICT tables."""
import sqlite3

db = sqlite3.connect(":memory:", autocommit=True)


def show(title, sql, params=()):
    print(f"\n-- {title}\n   {sql}")
    try:
        for row in db.execute(sql, params):
            print("  ", row)
    except sqlite3.Error as e:
        print("   ERROR:", type(e).__name__, e)


# 1. Column type is a hint ("affinity"), not a rule.
db.execute("CREATE TABLE loose (n INTEGER, s TEXT, b BLOB, x)")
db.execute("INSERT INTO loose VALUES ('42', 42, 42, 42)")       # convertible text -> integer
db.execute("INSERT INTO loose VALUES ('hello', 3.5, 'abc', 'abc')")  # not convertible -> kept as text
db.execute("INSERT INTO loose VALUES ('1e3', x'00ff', NULL, 1.0)")
show("Value stored vs declared type",
     "SELECT n, typeof(n), s, typeof(s), b, typeof(b), x, typeof(x) FROM loose")

# 2. Any type name is accepted; affinity is picked by substring rules.
db.execute("CREATE TABLE weird (a VARCHAR(3), b FLOATING POINT, c STRING, d BANANA, e DATETIME)")
db.execute("INSERT INTO weird VALUES ('way too long for 3', '12', '12', '12', '2024-01-01')")
show("VARCHAR(3) does NOT limit length; 'FLOATING POINT' -> INTEGER affinity (contains 'INT'); "
     "'STRING' -> NUMERIC affinity",
     "SELECT a, b, typeof(b), c, typeof(c), d, typeof(d), e, typeof(e) FROM weird")

# 3. Comparisons across storage classes: integers always sort before text.
db.execute("CREATE TABLE cmp (x)")  # no type -> BLOB affinity, no conversion
db.executemany("INSERT INTO cmp VALUES (?)", [(10,), ("9",), (2,), ("abc",), (None,), (b"\x01",)])
show("ORDER BY mixes classes: NULL < numbers < TEXT < BLOB", "SELECT x, typeof(x) FROM cmp ORDER BY x")
show("'9' (text) > 10 (integer) is TRUE", "SELECT '9' > 10, 9 > 10")
show("WHERE x = 9 does not match text '9' in a no-affinity column", "SELECT count(*) FROM cmp WHERE x = 9")

# 4. No real boolean or date types.
show("TRUE/FALSE are just 1/0", "SELECT TRUE, typeof(TRUE), FALSE")
show("Dates are text/number; functions do the work",
     "SELECT date('2024-01-31', '+1 month'), unixepoch('2024-01-01'), typeof(CURRENT_TIMESTAMP)")

# 5. Integer division and overflow.
show("Integer division truncates", "SELECT 7 / 2, 7 / 2.0, 7 % 3")
show("Integer overflow silently becomes REAL in arithmetic",
     "SELECT 9223372036854775807 + 1, typeof(9223372036854775807 + 1)")
show("...but sum() raises", "SELECT sum(x) FROM (SELECT 9223372036854775807 AS x UNION ALL SELECT 1)")

# 6. STRICT tables (3.37+) restore real type checking.
db.execute("CREATE TABLE strict_t (id INTEGER PRIMARY KEY, n INTEGER, s TEXT, any_col ANY) STRICT")
db.execute("INSERT INTO strict_t (n, s, any_col) VALUES ('42', 'ok', 'kept as text')")  # '42' converts losslessly
show("STRICT converts lossless '42' -> 42; ANY keeps exactly what you gave",
     "SELECT n, typeof(n), any_col, typeof(any_col) FROM strict_t")
show("STRICT rejects 'hello' in INTEGER column", "INSERT INTO strict_t (n) VALUES ('hello')")
show("STRICT rejects unknown type names",
     "CREATE TABLE bad (d DATETIME) STRICT")
