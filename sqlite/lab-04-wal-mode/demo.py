"""Lab 4 — WAL mode: readers don't block writers, snapshots, checkpoints, WAL growth."""
import os
import sqlite3
import tempfile
import threading
import time

# Real disk, not /tmp: /tmp is often tmpfs (RAM) where fsync is free and results lie.
_tmp = tempfile.TemporaryDirectory(dir=os.path.dirname(os.path.abspath(__file__)))
d = _tmp.name


def size(p):
    return os.path.getsize(p) if os.path.exists(p) else 0


def files(path):
    return f"db={size(path):>9,}  -wal={size(path + '-wal'):>9,}  -shm={size(path + '-shm'):>6,}"


def connect(path, timeout=0.0):
    return sqlite3.connect(path, autocommit=True, timeout=timeout, check_same_thread=False)


def reader_vs_writer(mode):
    """A reader holds an open transaction; can a writer commit?"""
    path = os.path.join(d, f"{mode}.db")
    w = connect(path)
    w.execute(f"PRAGMA journal_mode = {mode}")
    w.execute("CREATE TABLE t (v INT)")
    w.execute("INSERT INTO t VALUES (1)")
    r = connect(path)
    r.execute("BEGIN")
    before = r.execute("SELECT v FROM t").fetchone()
    try:
        w.execute("UPDATE t SET v = 2")
        result = "writer committed"
    except sqlite3.OperationalError as e:
        result = f"writer FAILED: {e}"
    still = r.execute("SELECT v FROM t").fetchone()
    r.execute("COMMIT")
    after = r.execute("SELECT v FROM t").fetchone()
    print(f"   {mode:6}: {result:40} reader saw {before} -> {still} inside txn, {after} after")


print("== 1. Long reader vs writer")
reader_vs_writer("DELETE")
reader_vs_writer("WAL")
print("   WAL: reader keeps its snapshot (v=1) until its transaction ends = snapshot isolation")

print("\n== 2. Throughput: 1 writer + 4 busy readers for 2 s")


def bench(mode):
    path = os.path.join(d, f"bench_{mode}.db")
    c = connect(path)
    c.execute(f"PRAGMA journal_mode = {mode}")
    c.execute("PRAGMA synchronous = NORMAL")
    c.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, v)")
    c.executemany("INSERT INTO t (v) VALUES (?)", [(i,) for i in range(10_000)])
    stop = time.perf_counter() + 2
    counts = {"writes": 0, "reads": 0, "busy": 0}
    lock = threading.Lock()

    def writer():
        w = connect(path, timeout=5)
        while time.perf_counter() < stop:
            w.execute("UPDATE t SET v = v + 1 WHERE id = 1")
            with lock:
                counts["writes"] += 1

    def reader():
        r = connect(path, timeout=5)
        while time.perf_counter() < stop:
            try:
                r.execute("SELECT sum(v) FROM t").fetchone()
                with lock:
                    counts["reads"] += 1
            except sqlite3.OperationalError:
                with lock:
                    counts["busy"] += 1

    ts = [threading.Thread(target=writer)] + [threading.Thread(target=reader) for _ in range(4)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    print(f"   {mode:6}: {counts}")


bench("DELETE")
bench("WAL")

print("\n== 3. Checkpoints and WAL growth")
path = os.path.join(d, "ckpt.db")
c = connect(path)
c.execute("PRAGMA journal_mode = WAL")
c.execute("CREATE TABLE t (blob)")
print("   wal_autocheckpoint (pages):", c.execute("PRAGMA wal_autocheckpoint").fetchone()[0])
for _ in range(50):
    c.execute("INSERT INTO t VALUES (randomblob(100000))")
print("   after 5 MB of inserts (autocheckpoint ran):", files(path))

# A reader holding an old snapshot blocks the checkpoint from finishing -> WAL keeps growing.
r = connect(path)
r.execute("BEGIN")
r.execute("SELECT count(*) FROM t").fetchone()
for _ in range(100):
    c.execute("INSERT INTO t VALUES (randomblob(100000))")
print("   10 MB more while a reader is stuck open:     ", files(path))
print("   checkpoint(PASSIVE) busy,log,done:", c.execute("PRAGMA wal_checkpoint(PASSIVE)").fetchone())
r.execute("COMMIT")
print("   checkpoint(TRUNCATE) after reader leaves: ", c.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone())
print("   ", files(path))

print("\n== 4. journal_mode=WAL is persistent (stored in the file)")
print("   new connection sees:", connect(path).execute("PRAGMA journal_mode").fetchone()[0])
