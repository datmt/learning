# Lab 07 — Production checklist (find slow + waste before users do)

**Question:** we're shipping — what do I check, and what do I set up so the
next slowdown pages *me* before users notice?

**Answer:** a 20-minute audit you can run on any Postgres: rank queries by
total time, drop unused indexes, find missing ones, check bloat, pre-compute
the heavy dashboard, partition the giant. The demo builds a small messy DB
(slow query + dead index + bloat + giant table) and finds every issue with
catalog queries you can steal for runbooks.

```bash
python3 demo.py              # default 200k-row events + partitioned sales
```

## 1. `pg_stat_statements` — where is time actually going?

```sql
SELECT left(query, 60), calls, round(total_exec_time::numeric) AS total_ms,
       round(mean_exec_time::numeric, 2) AS mean_ms
FROM pg_stat_statements
WHERE dbid = (SELECT oid FROM pg_database WHERE datname = current_database())
ORDER BY total_exec_time DESC LIMIT 10;
```

Sort by **total**, not mean: a 5 ms query × 1M calls hurts more than a 2 s
query × 3 calls. N+1 (lab 05) shows up as huge `calls`, tiny `mean`.
Reset after deploys to compare: `SELECT pg_stat_statements_reset()`.

## 2. Unused indexes — pure cost (lab 02 priced them)

```sql
SELECT indexrelname, pg_size_pretty(pg_relation_size(indexrelid))
FROM pg_stat_user_indexes
WHERE schemaname = 'public' AND idx_scan = 0 AND indexrelname NOT LIKE '%pkey%';
```

Caveats: check since last reset (`pg_stat_reset` / server start), keep
unique-constraint backers (they enforce correctness, not speed), and don't
drop the index that saves your monthly report. Demo creates a deliberately
useless index and catches it.

## 3. Missing indexes — tables that scream Seq Scan

```sql
SELECT relname, seq_scan, seq_tup_read, n_dead_tup
FROM pg_stat_user_tables WHERE schemaname='public' ORDER BY seq_tup_read DESC;
```

High `seq_scan` + huge `seq_tup_read` on a big table = reads scanning
everything repeatedly → candidate index (verify with `EXPLAIN`, labs 01–02).
High `n_dead_tup` = bloat → `VACUUM` / check for idle transactions (lab 06).

## 4. Guardrails to enable today

| Guardrail | What it does |
|---|---|
| `log_min_duration_statement = '500ms'` | slow-query log (already on in our compose) |
| `auto_explain` (`log_analyze`, `log_buffers`, `log_min_duration`) | logs the *plan* of slow queries, not just text |
| `log_temp_files = 0` | logs every disk spill (lab 05's cliff, now visible) |
| `statement_timeout` / `lock_timeout` / `idle_in_transaction_session_timeout` | lab 06's blast-radius bounds |
| PgBouncer (transaction pooling) | 500 app threads ≠ 500 Postgres backends; pool to ~2–4× cores |
| Alert on: connection count, replication lag, `n_dead_tup` growth, temp files | symptoms before outage |

## 5. Pre-compute + partition the heavy stuff

- **Materialized view** for the daily-revenue dashboard: demo shows the raw
  `GROUP BY` (~50 ms) vs `SELECT * FROM mv` (~0.3 ms), refreshed with
  `REFRESH MATERIALIZED VIEW CONCURRENTLY` (needs a unique index — demo
  includes it). Stale-by-minutes is fine for dashboards; realtime is for
  cash registers.
- **Declarative partitioning** (`PARTITION BY RANGE (month)`): demo's
  `EXPLAIN` shows *partition pruning* — a one-month query touches 1 of 3
  partitions. Partition when deletes are by time (drop a month = instant) or
  single partitions exceed RAM.

## 6. Ship checklist (copy into your PR template)

- [ ] Top-10 `pg_stat_statements` reviewed; new queries `EXPLAIN ANALYZE`d at SCALE.
- [ ] No new `OFFSET` without keyset (lab 04); no `SELECT` in a loop (lab 05).
- [ ] `ANALYZE` after migrations/backfills; autovacuum keeping up (`n_dead_tup` flat).
- [ ] Unused indexes dropped; index sizes recorded (`pg_relation_size`).
- [ ] Timeouts set per role; pool sized; slow-log + `auto_explain` on.
- [ ] Dashboard aggregates behind MV/rollup; time-series partitioned or BRIN'd.

## Exercises

1. `SELECT * FROM pg_stat_statements ORDER BY mean_exec_time DESC` — which
   query has the worst *mean*? Worst *total*? Different answers = the lesson.
2. Drop the demo's useless index, re-run §2. Empty = clean.
3. `EXPLAIN` the partitioned query with `enable_partition_pruning=off`.
   How many partitions now? (Pruning is load-bearing — don't disable it.)
