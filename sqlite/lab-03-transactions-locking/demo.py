"""Lab 3 — Transactions, locks, SQLITE_BUSY, busy_timeout (rollback-journal mode)."""
import os
import sqlite3
import tempfile
import threading
import time

path = os.path.join(tempfile.mkdtemp(), "lab3.db")


def connect(timeout=0.0):
    # autocommit=True: Python sends no hidden BEGIN; we control transactions with SQL.
    # timeout = busy_timeout in seconds. 0 = fail immediately when locked.
    return sqlite3.connect(path, autocommit=True, timeout=timeout, check_same_thread=False)


def attempt(label, conn, sql):
    try:
        conn.execute(sql)
        print(f"   {label}: OK  ({sql})")
    except sqlite3.OperationalError as e:
        print(f"   {label}: ERROR '{e}'  ({sql})")


setup = connect()
setup.execute("PRAGMA journal_mode = DELETE")  # classic rollback journal (the default)
setup.execute("CREATE TABLE acct (id INTEGER PRIMARY KEY, bal INT)")
setup.execute("INSERT INTO acct VALUES (1, 100), (2, 100)")

print("\n== 1. Autocommit: every statement is its own transaction (one fsync each)")
print("   in_transaction after INSERT:", setup.in_transaction)

print("\n== 2. A writer blocks readers in rollback-journal mode (at COMMIT time)")
a, b = connect(), connect()
a.execute("BEGIN")
a.execute("UPDATE acct SET bal = bal - 10 WHERE id = 1")   # RESERVED lock: readers still OK
attempt("B reads while A holds RESERVED", b, "SELECT * FROM acct")
attempt("B tries to write", b, "UPDATE acct SET bal = 0 WHERE id = 2")
b.execute("BEGIN")
b.execute("SELECT * FROM acct").fetchall()                   # B holds SHARED lock now
attempt("A commits while B is reading", a, "COMMIT")         # needs EXCLUSIVE -> blocked by B's SHARED
b.execute("COMMIT")
attempt("A commits after B finished", a, "COMMIT")

print("\n== 3. busy_timeout: wait instead of failing")
a = connect()
b = connect(timeout=3.0)
a.execute("BEGIN IMMEDIATE")
threading.Timer(0.5, lambda: a.execute("COMMIT")).start()   # A releases after 0.5 s
t0 = time.perf_counter()
attempt("B writes with 3 s busy_timeout", b, "UPDATE acct SET bal = bal + 1 WHERE id = 2")
print(f"   B waited {time.perf_counter() - t0:.2f}s")

print("\n== 4. The DEFERRED upgrade trap: busy_timeout cannot save you")
a, b = connect(timeout=3.0), connect(timeout=3.0)
a.execute("BEGIN")                  # DEFERRED: no lock yet
b.execute("BEGIN")
a.execute("SELECT * FROM acct").fetchall()   # A: SHARED
b.execute("SELECT * FROM acct").fetchall()   # B: SHARED
a.execute("UPDATE acct SET bal = 1 WHERE id = 1")  # A: SHARED -> RESERVED ok
t0 = time.perf_counter()
attempt("B upgrades read->write", b, "UPDATE acct SET bal = 2 WHERE id = 2")
print(f"   failed after {time.perf_counter() - t0:.2f}s (no waiting: waiting would deadlock)")
b.execute("ROLLBACK")
a.execute("COMMIT")

print("\n== 5. Fix: BEGIN IMMEDIATE takes the write lock up front, so busy_timeout works")
a, b = connect(timeout=3.0), connect(timeout=3.0)
a.execute("BEGIN IMMEDIATE")
threading.Timer(0.3, lambda: a.execute("COMMIT")).start()
t0 = time.perf_counter()
attempt("B BEGIN IMMEDIATE (waits)", b, "BEGIN IMMEDIATE")
print(f"   B got the lock after {time.perf_counter() - t0:.2f}s")
b.execute("SELECT * FROM acct").fetchall()
attempt("B writes", b, "UPDATE acct SET bal = 2 WHERE id = 2")
b.execute("COMMIT")

print("\n== 6. Atomicity: an error mid-transaction, then ROLLBACK")
c = connect()
c.execute("BEGIN")
c.execute("UPDATE acct SET bal = bal - 50 WHERE id = 1")
try:
    c.execute("INSERT INTO acct VALUES (1, 0)")   # PK conflict
except sqlite3.IntegrityError as e:
    print("   error:", e, "| still in transaction:", c.in_transaction)
    c.execute("ROLLBACK")
print("   balances:", c.execute("SELECT * FROM acct").fetchall())
print("   NOTE: a failed statement does NOT roll back the transaction by itself. You must.")

print("\n== 7. SAVEPOINT = nested transaction")
c.execute("BEGIN")
c.execute("UPDATE acct SET bal = 0 WHERE id = 1")
c.execute("SAVEPOINT sp")
c.execute("UPDATE acct SET bal = 0 WHERE id = 2")
c.execute("ROLLBACK TO sp")          # undo only id=2 change
c.execute("COMMIT")
print("   balances:", c.execute("SELECT * FROM acct").fetchall())
