# SQL Performance & Optimization — Hands-on Labs (PostgreSQL 16 + Docker)

Learn why queries get slow **at scale** and how to fix them — by measuring, not
guessing. Each lab = `README.md` (theory, 5–10 min) + `demo.py` (seeds its own
data, runs every claim, prints timings + plans). Labs are self-seeding and
re-runnable; run 00 → 07 in order the first time.

> Prerequisite: Docker + Python 3.10+. No local Postgres needed.
> This series uses its **own** Postgres 16 on port **5434** (so it won't clash
> with the `postgres-aggregation` labs on 5433 or a local 5432).

## Setup (once — the infra prerequisite)

```bash
cd postgres-perf

# 1. Start PostgreSQL 16 with pg_stat_statements + slow-query log pre-enabled
docker compose up -d
docker exec pg-perf-lab pg_isready -U perf -d perf   # expect "accepting connections"

# 2. Python driver (only dependency)
pip install -r requirements.txt   # psycopg2-binary

# 3. Verify infra + build your measurement habit (start here)
cd lab-00-infra-setup && python3 demo.py
```

Connection defaults (override with env vars):

| Var | Default |
|---|---|
| `PGHOST` | `localhost` |
| `PGPORT` | `5434` |
| `PGDATABASE` | `perf` |
| `PGUSER` | `perf` |
| `PGPASSWORD` | `perf_pw` |
| `SCALE` | lab-dependent (e.g. `200000` rows; raise to `1000000` to feel real pain) |

Every `demo.py` does `DROP TABLE IF EXISTS` + `CREATE TABLE` + deterministic
seed (mostly server-side `generate_series`, so seeding 200k rows takes ~1–2 s),
then prints proof. Re-running is always safe.

```bash
# Talk to the DB directly
docker exec -it pg-perf-lab psql -U perf -d perf
# or from the host (needs local psql):
PGPASSWORD=perf_pw psql -h localhost -p 5434 -U perf -d perf
```

```bash
# Stop / wipe everything
docker compose down        # stop, keep data
docker compose down -v     # stop AND delete all seeded data
```

## Lab map

| Lab | Question it answers | Key SQL / concepts |
|---|---|---|
| [00 infra + measuring](lab-00-infra-setup/) | Is my DB up? How do I time honestly (cold vs warm, `EXPLAIN ANALYZE, BUFFERS`)? | Docker, `pg_isready`, `EXPLAIN (ANALYZE, BUFFERS, TIMING)`, `pg_stat_statements`, `SCALE` |
| [01 reading query plans](lab-01-query-plan-reading/) | What is the planner telling me? Seq Scan vs Index Scan, Nested Loop vs Hash Join, bad row estimates? | `EXPLAIN`, cost/rows vs actual, `Bitmap Heap Scan`, `Sort`, `Limit`, `ANALYZE` + stats |
| [02 indexing fundamentals](lab-02-indexing-fundamentals/) | Which index fixes a slow `WHERE`? Why didn't mine get used? Composite order? | B-tree, selectivity, leftmost prefix, covering `INCLUDE`, index-only scans, write cost |
| [03 advanced indexes](lab-03-advanced-indexes/) | Partial? Expression? Full-text / JSONB? Time-series at 10M rows? | `WHERE` partial, `lower()` expression, `GIN`, `BRIN`, bloat, `REINDEX` |
| [04 paging at scale](lab-04-paging-at-scale/) | Why is page 50,000 slow? How do infinite-scroll APIs stay fast? `COUNT(*)` for "page 1 of N"? | `OFFSET` O(N) failure, keyset `WHERE id > $1`, deferred join, cursor, estimate counts |
| [05 joins & aggregations](lab-05-joins-aggregations-scale/) | Why did my join explode? `GROUP BY` spills to disk? `IN` vs `EXISTS`? | Join algorithms, `work_mem` + temp files, `EXISTS`, CTE materialization, denormalization |
| [06 writes, locks & failure modes](lab-06-writes-locks-failure-modes/) | Deadlock? `database is locked`? One slow query blocks deploy? Thundering herd? | Row locks, `FOR UPDATE SKIP LOCKED` queue, idle-in-txn blocking `VACUUM`, `statement_timeout`, connection storm |
| [07 production checklist](lab-07-production-checklist/) | What do I check before / after shipping? Which indexes are dead weight? | `pg_stat_statements`, unused indexes, bloat, `auto_explain`, pooling, materialized views, partitioning |

## Mental model (read once)

1. **Measure, don't guess.** Every optimization in this series is proven with
   `EXPLAIN (ANALYZE, BUFFERS)` timings before/after. If you can't measure it,
   don't ship it.
2. **Scale changes the answer.** At 1k rows every plan is fast; at 1M rows a
   Seq Scan is 1000× slower but an index lookup is ~same. All scale labs take
   `SCALE` so you can turn the pain up.
3. **The planner runs on statistics, not truth.** `ANALYZE` builds histograms;
   stale stats → wrong row estimates → wrong plan (lab 01). `EXPLAIN` shows
   *estimated* cost; `EXPLAIN ANALYZE` shows *actual* time+rows — compare them.
4. **Indexes are a trade.** Each index speeds some reads and taxes every write
   (+ disk). Unused indexes are pure cost (lab 07 shows how to find them).
5. **Failure modes are load-dependent.** OFFSET, N+1, missing index, open
   transaction — all "fine on laptop, outage in prod". Each lab has a failure
   section that reproduces the prod shape at small scale.
