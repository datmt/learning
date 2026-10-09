# Lab 00 — Infra setup + how to measure (prerequisite for everything)

**Question:** is my Postgres 16 actually up, and how do I time a query honestly?

**Answer:** this lab is the foundation. It verifies Docker → Postgres 16 →
`pg_stat_statements`, shows the settings that affect every later lab
(`shared_buffers`, `work_mem`), and teaches the one habit that matters:
**`EXPLAIN (ANALYZE, BUFFERS)` before and after every change.**

```bash
# from postgres-perf/ : docker compose up -d   (once)
python3 demo.py              # needs PG* env from top-level README
SCALE=200000 python3 demo.py # bigger seed, default 50000 (kept small here)
```

## 1. What `docker compose up -d` gave you

- Image `postgres:16-alpine`, container `pg-perf-lab`, DB `perf` on host port **5434**.
- `shared_preload_libraries=pg_stat_statements` + `pg_stat_statements.track=all`,
  so lab 07 can rank queries by total time.
- `log_min_duration_statement=500` — any query slower than 500 ms is logged;
  watch with `docker logs pg-perf-lab`.

Check it yourself: `docker exec pg-perf-lab pg_isready -U perf -d perf`.

## 2. Settings that shape every benchmark

| Setting | What it does | Demo default |
|---|---|---|
| `shared_buffers` | Postgres's own page cache (128 MB in Docker default) | read-only in demo |
| `work_mem` | Sort/hash memory *per operation* before spilling to disk (lab 05) | 4 MB |
| `random_page_cost` | Planner's guess how expensive non-sequential reads are (SSD: lower to ~1.1) | 4.0 (default, HDD-era) |
| `effective_cache_size` | Planner's guess of total cache (OS + shared) — affects index-vs-scan choice | 4 GB |

The demo prints all four. Don't tune them yet — just know they exist.

## 3. The measurement habit (use in every later lab)

```sql
EXPLAIN (ANALYZE, BUFFERS, TIMING)
SELECT ... ;   -- your query
```

- `EXPLAIN` alone = planner's **estimate** (cost, estimated rows). Cheap, never runs.
- `EXPLAIN ANALYZE` = actually **runs** it, shows real time + real rows.
  Compare `rows=100 (est)` vs `rows=98000 (actual)` — a 1000× misestimate
  means stale stats (lab 01).
- `BUFFERS` adds `shared hit / read`, `temp read / written`.
  `hit` = found in cache (fast). `read` = from disk (slow first time).
  `temp written` = spilled to disk (lab 05's `work_mem` story).

Rules: run each query **twice** (cold = first, warm = cached), use
`pg_stat_statements` for production ranking (single timings lie), and seed
with `generate_series` (server-side, fast) instead of Python loops.

## 4. What the demo proves

1. Server version is 16.x, `pg_stat_statements` installed + tracking.
2. Prints `shared_buffers / work_mem / random_page_cost / effective_cache_size`.
3. Seeds `SCALE` rows into `lab00_events` with `generate_series` (~1 s per 100k).
4. Same `COUNT WHERE status='rare'` three ways: naive timing vs
   `EXPLAIN ANALYZE` vs `BUFFERS` — cold run shows `read`, rerun shows `hit`.
5. Queries `pg_stat_statements` to prove statements are being tracked.

## Exercises

1. Run with `SCALE=10000` then `SCALE=500000`. Does seed time scale linearly?
   Does the cold `read` count grow while warm time stays flat?
2. `docker logs pg-perf-lab | tail -20` — trigger a > 500 ms query
   (`SELECT count(*) FROM lab00_events` twice won't do it; try a big sort in lab 05)
   and find it in the log.
3. In `psql`, run `SHOW work_mem;` then `SET work_mem='1MB';` — session-only.
   Lab 05 will use this to force a disk spill on demand.
