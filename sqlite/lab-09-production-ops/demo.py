"""Lab 9 — Production operations: backups, file size/VACUUM, integrity checks, optimize."""
import os
import shutil
import sqlite3
import tempfile

d = tempfile.mkdtemp()
path = os.path.join(d, "app.db")


def size(p):
    return os.path.getsize(p) if os.path.exists(p) else 0


def mb(p):
    return f"{size(p) / 1e6:6.2f} MB"


db = sqlite3.connect(path, autocommit=True)
db.execute("PRAGMA journal_mode = WAL")
db.execute("PRAGMA wal_autocheckpoint = 0")   # keep data in the WAL to make part 1 obvious
db.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, payload BLOB)")
db.execute("BEGIN")
db.executemany("INSERT INTO t (payload) VALUES (randomblob(1000))", [()] * 10_000)
db.execute("COMMIT")
print(f"== 0. app.db {mb(path)}, app.db-wal {mb(path + '-wal')}  (10,000 committed rows)")

print("\n== 1. WRONG backup: copy only the .db file while the app runs")
shutil.copy(path, os.path.join(d, "naive.db"))
try:
    n = sqlite3.connect(os.path.join(d, "naive.db")).execute("SELECT count(*) FROM t").fetchone()[0]
    print(f"   rows in copy: {n}")
except sqlite3.Error as e:
    print(f"   copy is broken: {e}")
print("   committed data lives in -wal until checkpoint; copying db alone loses it")
print("   (copying db + wal separately while writes happen can also give a corrupt pair)")

print("\n== 2. RIGHT backup #1: online backup API (consistent snapshot, app keeps running)")
dst = sqlite3.connect(os.path.join(d, "backup_api.db"))
db.backup(dst, pages=1000)   # copy in steps of 1000 pages; other connections can write between steps
print("   rows:", dst.execute("SELECT count(*) FROM t").fetchone()[0])
dst.close()

print("\n== 3. RIGHT backup #2: VACUUM INTO (consistent + compacted copy)")
db.execute("VACUUM INTO ?", (os.path.join(d, "vacuum_into.db"),))
print("   rows:", sqlite3.connect(os.path.join(d, "vacuum_into.db")).execute("SELECT count(*) FROM t").fetchone()[0])

db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
db.execute("PRAGMA wal_autocheckpoint = 1000")

print("\n== 4. DELETE does not shrink the file (pages go to the freelist)")
print(f"   before delete: {mb(path)}  freelist pages={db.execute('PRAGMA freelist_count').fetchone()[0]}")
db.execute("DELETE FROM t WHERE id % 10 != 0")   # delete 90%
db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
print(f"   after delete:  {mb(path)}  freelist pages={db.execute('PRAGMA freelist_count').fetchone()[0]}")
db.execute("VACUUM")
db.execute("PRAGMA wal_checkpoint(TRUNCATE)")   # in WAL mode the rewritten pages sit in -wal first
print(f"   after VACUUM:  {mb(path)}  freelist pages={db.execute('PRAGMA freelist_count').fetchone()[0]}")
print("   VACUUM rewrites the whole db: needs exclusive lock + up to 2x disk space")

print("\n== 5. auto_vacuum = INCREMENTAL: reclaim space in small steps, no full rewrite")
p2 = os.path.join(d, "inc.db")
c = sqlite3.connect(p2, autocommit=True)
c.execute("PRAGMA auto_vacuum = INCREMENTAL")   # must be set before first table (or followed by VACUUM)
c.execute("CREATE TABLE t (b BLOB)")
c.execute("BEGIN")
c.executemany("INSERT INTO t VALUES (randomblob(1000))", [()] * 10_000)
c.execute("COMMIT")
c.execute("DELETE FROM t")
print(f"   after delete: {mb(p2)}  free={c.execute('PRAGMA freelist_count').fetchone()[0]}")
c.execute("PRAGMA incremental_vacuum(1000)")   # QUIRK: frees 1 page per step of the statement
print(f"   incremental_vacuum(1000), not stepped: {mb(p2)}  free={c.execute('PRAGMA freelist_count').fetchone()[0]}")
c.execute("PRAGMA incremental_vacuum(1000)").fetchall()   # step to the end -> frees 1000 pages
print(f"   incremental_vacuum(1000) + fetchall(): {mb(p2)}  free={c.execute('PRAGMA freelist_count').fetchone()[0]}")

print("\n== 6. Integrity checks")
print("   quick_check:", db.execute("PRAGMA quick_check").fetchone()[0])
print("   integrity_check:", db.execute("PRAGMA integrity_check").fetchone()[0])
db.close()
# Simulate disk corruption: overwrite bytes in the middle of the file.
bad = os.path.join(d, "corrupt.db")
shutil.copy(path, bad)
with open(bad, "r+b") as f:
    f.seek(4096 * 3)
    f.write(os.urandom(4096))
c = sqlite3.connect(bad)
try:
    rows = c.execute("PRAGMA integrity_check").fetchall()
    print("   corrupted file:", rows[:3], f"... ({len(rows)} problems)" if len(rows) > 3 else "")
except sqlite3.DatabaseError as e:
    print("   corrupted file:", e)

print("\n== 7. PRAGMA optimize: cheap auto-ANALYZE, run at connection close / periodically")
c = sqlite3.connect(path, autocommit=True)
c.execute("CREATE INDEX IF NOT EXISTS ix ON t(payload)")
c.execute("SELECT * FROM t WHERE payload = x'00'").fetchall()
print("   would run:", c.execute("PRAGMA optimize(-1)").fetchall() or "(nothing)")
c.execute("PRAGMA optimize")
print("   sqlite_stat1 rows after optimize:", c.execute("SELECT count(*) FROM sqlite_stat1").fetchone()[0])

print("\n== 8. user_version: built-in slot for schema migration number")
print("   user_version before:", c.execute("PRAGMA user_version").fetchone()[0])
MIGRATIONS = ["ALTER TABLE t ADD COLUMN note TEXT", "CREATE INDEX ix_note ON t(note)"]
ver = c.execute("PRAGMA user_version").fetchone()[0]
for i, sql in enumerate(MIGRATIONS[ver:], start=ver + 1):
    c.execute("BEGIN IMMEDIATE")
    c.execute(sql)
    c.execute(f"PRAGMA user_version = {i}")   # DDL + version bump are atomic together
    c.execute("COMMIT")
print("   user_version after:", c.execute("PRAGMA user_version").fetchone()[0])

shutil.rmtree(d)
