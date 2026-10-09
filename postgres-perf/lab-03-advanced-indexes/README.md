# Lab 03 — Advanced indexes (partial, expression, GIN, BRIN)

**Question:** B-tree on the column didn't help — what's next?

**Answer:** pick the index that matches the *query shape*, not the column:
hot-subset queries → **partial** index; function/`LOWER()` queries →
**expression** index; JSONB / full-text / arrays → **GIN**; append-only
time-series → **BRIN** (1000× smaller). Each demo below shows the failing
B-tree plan first, then the fix.

```bash
python3 demo.py              # default 200k rows
SCALE=500000 python3 demo.py
```

## 1. Partial index — index the hot subset

```sql
CREATE INDEX ON orders (customer_id) WHERE status = 'open';
SELECT ... WHERE status='open' AND customer_id=42;   -- uses it
SELECT ... WHERE status='closed' AND customer_id=42; -- correctly doesn't
```

Why: 2% of rows are `open` but 95% of queries hunt them. The partial index is
~50× smaller, faster to scan, cheaper to maintain — cold rows' writes don't
touch it at all. Predicate must **match exactly** (`status='open'` in both).

## 2. Expression index — index the computed value

Lab 02 proved `WHERE lower(email)=...` ignores a plain index. Fix:

```sql
CREATE INDEX ON users (lower(email));
```

Now the planner matches `lower(email)` in query to `lower(email)` in index.
Same trick for `date_trunc('day', ts)`, `(total - discount)`, JSONB
`->>` extractions. Cost: expression evaluated on every write.

## 3. GIN — JSONB, full-text, arrays

B-trees index *scalar* values; GIN indexes *elements inside* a value:

| Query | Index |
|---|---|
| `payload @> '{"vip": true}'` (JSONB contains) | `USING gin (payload)` or `jsonb_path_ops` (smaller, only `@>`) |
| `to_tsvector('english', body) @@ to_tsquery('refund & delay')` | `USING gin (to_tsvector('english', body))` |
| `tags @> ARRAY['sale']` | `USING gin (tags)` |
| `email LIKE '%gmail%'` (substring!) | `USING gin (email gin_trgm_ops)` + `pg_trgm` extension |

GIN is bigger and slower to update than B-tree (pending list) — great for
read-heavy search, bad for write-heavy counters. Demo measures all three.

## 4. BRIN — time-series at scale

For append-only tables (metrics, events, logs) where `created_at` grows with
physical row order, BRIN stores one min/max per page block instead of one entry
per row: **~180× smaller** than B-tree (`24 kB` vs `~4.4 MB` at 200k rows in the
demo). Range queries (`WHERE ts BETWEEN ...`) skip whole blocks. Useless on
random-order columns (UUIDs) — min/max per block overlaps everything.

## 5. Bloat + REINDEX (know it exists)

Updates/deletes leave dead entries; `VACUUM` reclaims space but index pages stay
half-empty. Symptoms: index 3× bigger than a fresh build. Fix: `REINDEX
CONCURRENTLY` (no lock in PG12+). Demo shows size before/after churn.

## Exercises

1. Change the partial predicate to `WHERE total > 1000` — which queries use it?
   What happens if the query says `total >= 1000` (mismatch)?
2. `EXPLAIN` the `@@` query without the GIN index. Seq Scan + filter on
   `to_tsvector` — how slow at your SCALE?
3. Insert 50k rows with random `created_at` into the metrics table and re-run
   the BRIN range query. When does BRIN stop helping?
