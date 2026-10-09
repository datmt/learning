"""Lab 5 — Write performance: transactions, synchronous, journal mode, batching."""
import os
import sqlite3
import tempfile
import time

# Real disk, not /tmp: /tmp is often tmpfs (RAM) where fsync is free and results lie.
_tmp = tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(__file__)))
d = _tmp.name
N = 2_000
ROWS = [(i, f"user{i}", i * 1.5) for i in range(N)]


def run(label, journal, sync, batch, rows=ROWS):
    path = os.path.join(d, f"{label}.db".replace(" ", "_"))
    c = sqlite3.connect(path, autocommit=True)
    c.execute(f"PRAGMA journal_mode = {journal}")
    c.execute(f"PRAGMA synchronous = {sync}")
    c.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, score REAL)")
    t0 = time.perf_counter()
    if batch:
        c.execute("BEGIN")
        c.executemany("INSERT INTO t VALUES (?, ?, ?)", rows)
        c.execute("COMMIT")
    else:
        for r in rows:
            c.execute("INSERT INTO t VALUES (?, ?, ?)", r)   # autocommit: 1 txn per row
    dt = time.perf_counter() - t0
    print(f"   {label:38} {len(rows):>9,} rows  {dt:7.3f}s  {len(rows) / dt:>12,.0f} rows/s")
    c.close()


print(f"== 1. {N:,} inserts, one transaction per row vs one transaction total")
run("DELETE  sync=FULL   row-by-row", "DELETE", "FULL", False)
run("WAL     sync=FULL   row-by-row", "WAL", "FULL", False)
run("WAL     sync=NORMAL row-by-row", "WAL", "NORMAL", False)
run("WAL     sync=OFF    row-by-row", "WAL", "OFF", False)
run("DELETE  sync=FULL   one txn", "DELETE", "FULL", True)
run("WAL     sync=NORMAL one txn", "WAL", "NORMAL", True)

print("\n== 2. Bulk load 1,000,000 rows in one transaction")
big = [(i, f"user{i}", i * 1.5) for i in range(1_000_000)]
run("WAL sync=NORMAL executemany", "WAL", "NORMAL", True, big)

print("\n== 3. Index before vs after bulk load")
for when in ("before", "after"):
    path = os.path.join(d, f"idx_{when}.db")
    c = sqlite3.connect(path, autocommit=True)
    c.execute("PRAGMA journal_mode = WAL")
    c.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, name TEXT, score REAL)")
    t0 = time.perf_counter()
    if when == "before":
        c.execute("CREATE INDEX ix ON t(name)")
    c.execute("BEGIN")
    c.executemany("INSERT INTO t VALUES (?, ?, ?)", big)
    c.execute("COMMIT")
    if when == "after":
        c.execute("CREATE INDEX ix ON t(name)")
    print(f"   index created {when:6} load: {time.perf_counter() - t0:.3f}s")
    c.close()

print("\n== 4. Random UUID-like keys vs sequential keys (B-tree page splits)")
import uuid
for kind in ("sequential", "random"):
    path = os.path.join(d, f"key_{kind}.db")
    c = sqlite3.connect(path, autocommit=True)
    c.execute("CREATE TABLE t (k TEXT PRIMARY KEY, v INT) WITHOUT ROWID")
    keys = [f"{i:032d}" for i in range(300_000)] if kind == "sequential" else [uuid.uuid4().hex for _ in range(300_000)]
    t0 = time.perf_counter()
    c.execute("BEGIN")
    c.executemany("INSERT INTO t VALUES (?, 0)", ((k,) for k in keys))
    c.execute("COMMIT")
    dt = time.perf_counter() - t0
    print(f"   {kind:10} keys: {dt:.3f}s  file={os.path.getsize(path):,} bytes")
    c.close()
