"""Lab 06 — live lock/block/deadlock/queue/idle-txn/statement-timeout demos."""
import os
import threading
import time
import collections
import psycopg2

CFG = dict(
    host=os.getenv("PGHOST", "localhost"),
    port=int(os.getenv("PGPORT", "5434")),
    dbname=os.getenv("PGDATABASE", "perf"),
    user=os.getenv("PGUSER", "perf"),
    password=os.getenv("PGPASSWORD", "perf_pw"),
)


def conn():
    c = psycopg2.connect(**CFG)
    c.autocommit = True
    return c


def main():
    admin = conn()
    cur = admin.cursor()
    cur.execute("DROP TABLE IF EXISTS lab06_accounts")
    cur.execute("DROP TABLE IF EXISTS lab06_jobs")
    cur.execute("CREATE TABLE lab06_accounts (id INT PRIMARY KEY, balance NUMERIC)")
    cur.execute("INSERT INTO lab06_accounts VALUES (1, 100), (2, 100)")
    cur.execute("CREATE TABLE lab06_jobs (id INT PRIMARY KEY, status TEXT)")
    cur.execute("INSERT INTO lab06_jobs SELECT g, 'queued' FROM generate_series(1, 20) g")
    print("seeded lab06_accounts (2 rows) + lab06_jobs (20 queued)")

    print("\n== 1. blocked UPDATE + lock_timeout (fail fast, don't wedge the pool) ==")
    a = psycopg2.connect(**CFG)  # autocommit OFF: explicit txn
    a.cursor().execute("BEGIN")
    a.cursor().execute("UPDATE lab06_accounts SET balance=balance-10 WHERE id=1")
    print("   conn A: BEGIN + UPDATE id=1 (holding row lock, staying open...)")
    b = conn()
    b.cursor().execute("SET lock_timeout='2s'")
    t0 = time.perf_counter()
    try:
        b.cursor().execute("UPDATE lab06_accounts SET balance=balance+10 WHERE id=1")
        print("   conn B: updated?! (unexpected — A should still hold the lock)")
    except psycopg2.errors.LockNotAvailable as e:
        print(f"   conn B: lock_timeout fired after {time.perf_counter()-t0:.1f}s "
              f"-> {e.pgcode} {str(e).strip().splitlines()[0][:80]}")
    a.cursor().execute("COMMIT")
    print("   conn A: COMMIT (lock released)")
    b.close()
    a.close()

    print("\n== 2. deadlock: opposite lock order (victim must be retried) ==")
    results = {}

    def transfer(name, first, second):
        c = psycopg2.connect(**CFG)
        c.autocommit = False
        try:
            cu = c.cursor()
            cu.execute("UPDATE lab06_accounts SET balance=balance-1 WHERE id=%s", (first,))
            time.sleep(0.5)  # let the other side grab its first lock -> guaranteed cycle
            cu.execute("UPDATE lab06_accounts SET balance=balance+1 WHERE id=%s", (second,))
            c.commit()
            results[name] = "committed"
        except psycopg2.errors.DeadlockDetected as e:
            c.rollback()
            results[name] = f"DEADLOCK victim ({e.pgcode}) -> app should retry"
        except Exception as e:  # noqa: BLE001
            c.rollback()
            results[name] = f"other error: {type(e).__name__} {e}"
        finally:
            c.close()

    t1 = threading.Thread(target=transfer, args=("A: 1->2", 1, 2))
    t2 = threading.Thread(target=transfer, args=("B: 2->1", 2, 1))
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    for k, v in results.items():
        print(f"   {k}: {v}")
    assert any("DEADLOCK" in v for v in results.values()), "expected one deadlock victim"

    print("\n== 3. queue: 4 workers, SKIP LOCKED, each job once ==")
    cur.execute("UPDATE lab06_jobs SET status='queued'")
    claimed = collections.Counter()
    lock = threading.Lock()

    def worker(w):
        c = conn()
        while True:
            cu = c.cursor()
            cu.execute("BEGIN")
            cu.execute("""SELECT id FROM lab06_jobs WHERE status='queued'
                          ORDER BY id LIMIT 3 FOR UPDATE SKIP LOCKED""")
            rows = cu.fetchall()
            if not rows:
                cu.execute("COMMIT")
                break
            ids = [r[0] for r in rows]
            cu.execute("UPDATE lab06_jobs SET status='done' WHERE id = ANY(%s)", (ids,))
            cu.execute("COMMIT")
            with lock:
                for i in ids:
                    claimed[i] += 1
        c.close()

    threads = [threading.Thread(target=worker, args=(w,)) for w in range(4)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    cur.execute("SELECT status, count(*) FROM lab06_jobs GROUP BY 1")
    print(f"   final: {dict(cur.fetchall())}, claimed rows={len(claimed)}, "
          f"double-claimed={sum(1 for v in claimed.values() if v > 1)} (expect 0)")

    print("\n== 4. idle-in-transaction pins VACUUM (watch pg_stat_activity) ==")
    holder = psycopg2.connect(**CFG)
    holder.autocommit = False
    holder.cursor().execute("SELECT count(*) FROM lab06_accounts")
    cur.execute("""SELECT pid, state, now() - xact_start AS xact_age
                   FROM pg_stat_activity
                   WHERE datname='perf' AND state='idle in transaction'
                   ORDER BY xact_start LIMIT 3""")
    rows = cur.fetchall()
    print(f"   idle-in-transaction sessions: {len(rows)}")
    for pid, state, age in rows:
        print(f"     pid={pid} state={state} xact_age={age} <- blocks VACUUM, bloats tables")
    holder.rollback()
    holder.close()
    print("   rolled back. Guard: idle_in_transaction_session_timeout='30s'.")

    print("\n== 5. statement_timeout bounds the blast radius ==")
    s = conn()
    s.cursor().execute("SET statement_timeout='200ms'")
    t0 = time.perf_counter()
    try:
        s.cursor().execute("SELECT pg_sleep(1)")
    except psycopg2.errors.QueryCanceled as e:
        print(f"   pg_sleep(1) canceled after {time.perf_counter()-t0:.2f}s "
              f"-> {e.pgcode} (query_canceled). Pool survived.")
    s.close()

    print("\nDone. Takeaway: short txns, ordered locks, retry 40P01/55P03, "
          "SKIP LOCKED queues, always set the three timeouts.")
    cur.close()
    admin.close()


if __name__ == "__main__":
    main()
