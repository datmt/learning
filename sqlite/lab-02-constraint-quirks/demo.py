"""Lab 2 — Constraint and SQL quirks that surprise people coming from Postgres/MySQL."""
import sqlite3

db = sqlite3.connect(":memory:", autocommit=True)


def show(title, sql, params=()):
    print(f"\n-- {title}\n   {sql}")
    try:
        for row in db.execute(sql, params):
            print("  ", row)
    except sqlite3.Error as e:
        print("   ERROR:", type(e).__name__, e)


# 1. Foreign keys are OFF by default, per connection.
db.executescript("""
CREATE TABLE author (id INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE book (id INTEGER PRIMARY KEY, author_id INTEGER REFERENCES author(id));
""")
show("foreign_keys default", "PRAGMA foreign_keys")
show("Orphan insert succeeds when FK is off", "INSERT INTO book (author_id) VALUES (999)")
show("orphans", "SELECT * FROM book")
db.execute("PRAGMA foreign_keys = ON")
show("Same insert with FK on", "INSERT INTO book (author_id) VALUES (998)")
show("Find existing violations", "PRAGMA foreign_key_check")

# 2. rowid and INTEGER PRIMARY KEY.
db.execute("CREATE TABLE r (name TEXT)")
db.execute("INSERT INTO r VALUES ('a'), ('b')")
show("Every table has a hidden rowid", "SELECT rowid, name FROM r")
db.execute("CREATE TABLE alias (id INTEGER PRIMARY KEY, v)")
db.execute("CREATE TABLE notalias (id INT PRIMARY KEY, v)")
show("Only exactly 'INTEGER PRIMARY KEY' aliases rowid",
     "SELECT name, sql FROM sqlite_schema WHERE name LIKE 'sqlite_autoindex%' OR name IN ('alias','notalias')")

# 3. Reused ids without AUTOINCREMENT.
db.execute("CREATE TABLE plain (id INTEGER PRIMARY KEY)")
db.execute("CREATE TABLE auto (id INTEGER PRIMARY KEY AUTOINCREMENT)")
for t in ("plain", "auto"):
    db.execute(f"INSERT INTO {t} VALUES (NULL), (NULL), (NULL)")
    db.execute(f"DELETE FROM {t} WHERE id = 3")
    db.execute(f"INSERT INTO {t} VALUES (NULL)")
show("plain reuses max id 3; AUTOINCREMENT never reuses",
     "SELECT 'plain', max(id) FROM plain UNION ALL SELECT 'auto', max(id) FROM auto")

# 4. NULL allowed in a non-integer PRIMARY KEY (legacy bug kept for compatibility).
db.execute("CREATE TABLE pk_text (code TEXT PRIMARY KEY)")
db.execute("INSERT INTO pk_text VALUES (NULL), (NULL)")
show("Two NULL primary keys!", "SELECT count(*) FROM pk_text WHERE code IS NULL")
db.execute("CREATE TABLE pk_text_strict (code TEXT PRIMARY KEY) STRICT")
show("STRICT fixes it", "INSERT INTO pk_text_strict VALUES (NULL)")

# 5. UNIQUE allows many NULLs (same as Postgres default, unlike SQL Server).
db.execute("CREATE TABLE u (email TEXT UNIQUE)")
db.execute("INSERT INTO u VALUES (NULL), (NULL)")
show("UNIQUE + multiple NULLs ok", "SELECT count(*) FROM u")

# 6. Double-quoted strings fall back to string literals if no such column.
db.execute("CREATE TABLE q (name TEXT)")
db.execute("INSERT INTO q VALUES ('alice')")
show("Typo'd column in double quotes becomes a string literal (no error!)",
     'SELECT * FROM q WHERE "nmae" = \'nmae\'')
show("Single quotes = string, double quotes = identifier. Correct:", "SELECT * FROM q WHERE \"name\" = 'alice'")

# 7. LIKE is case-insensitive for ASCII only; GLOB is case-sensitive.
show("LIKE ASCII case-insensitive, but not for non-ASCII",
     "SELECT 'ABC' LIKE 'abc', 'ÄBC' LIKE 'äbc', 'ABC' GLOB 'abc', lower('Ä')")
show("= is case-sensitive unless COLLATE NOCASE", "SELECT 'a' = 'A', 'a' = 'A' COLLATE NOCASE")

# 8. Bare columns in aggregate queries are allowed (MySQL-style), with a useful special case.
db.execute("CREATE TABLE sale (who TEXT, amount INT)")
db.execute("INSERT INTO sale VALUES ('ann', 5), ('bob', 50), ('cid', 20)")
show("With max(), the bare column comes from the max row", "SELECT who, max(amount) FROM sale")
show("Without min/max, bare column value is arbitrary", "SELECT who, sum(amount) FROM sale")

# 9. ALTER TABLE is limited.
show("Can't add a column with non-constant default",
     "ALTER TABLE sale ADD COLUMN at TEXT DEFAULT CURRENT_TIMESTAMP")
show("Can't add a PRIMARY KEY/UNIQUE column", "ALTER TABLE sale ADD COLUMN k INT UNIQUE")
show("DROP COLUMN works since 3.35", "ALTER TABLE sale DROP COLUMN amount")
show("after drop", "SELECT * FROM sale")
