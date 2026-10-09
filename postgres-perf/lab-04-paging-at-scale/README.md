# Lab 04 — Paging at scale (`OFFSET` vs keyset)

**Question:** page 1 is instant, page 50,000 times out — why, and what do
infinite-scroll APIs do instead?

**Answer:** `OFFSET N LIMIT 20` still **reads + sorts + discards N rows** on
every request — O(N) per page, worse the deeper you go. Keyset pagination
(`WHERE id > last_seen ORDER BY id LIMIT 20`) seeks straight to the position
— O(1) per page. The demo times page 1 vs page 100,000 both ways on 500k rows.

```bash
python3 demo.py              # default 500k rows (~3 s seed)
SCALE=1000000 python3 demo.py
```

## 1. Why OFFSET is O(N) — read the plan

```
EXPLAIN ANALYZE SELECT ... ORDER BY id LIMIT 20 OFFSET 400000;
-> Limit (actual rows=20) over Sort/Index Scan that scanned 400020 rows
```

Postgres can't "jump" to offset 400k; it walks 400k rows and throws them away.
Deeper page = more discarded work = slower + more `BUFFERS hit`. Under
concurrent load this is a classic outage shape: bots crawl deep pages, every
request scans half the table.

## 2. Keyset (seek method) — the standard fix

```sql
-- page 1
SELECT id, created_at FROM items ORDER BY id LIMIT 20;
-- next page (client sends back last id=1020)
SELECT id, created_at FROM items WHERE id > 1020 ORDER BY id LIMIT 20;
```

An index on `(id)` seeks to 1020 and reads 20 rows — page 1 and page 100,000
cost the same (~0.05 ms). Requirements: a **unique, monotonic** ordering key
(`id`, or `(created_at, id)` tiebreak for timestamp feeds). No arbitrary page
jumps ("go to page 842") — that's the trade; most feeds don't need it.

For timestamp feeds with ties:

```sql
WHERE (created_at, id) > ($last_ts, $last_id)
ORDER BY created_at, id LIMIT 20;
```

Needs index `(created_at, id)`.

## 3. Deferred join — damage control when you must keep page numbers

If the UI needs page numbers, at least avoid fetching wide rows for discarded
offsets: scan ids on a covering index, then join back for 20 rows:

```sql
SELECT t.* FROM t JOIN (
  SELECT id FROM t ORDER BY id LIMIT 20 OFFSET 400000
) p USING (id) ORDER BY id;
```

Honest caveat (measured in the demo): with narrow rows this barely helps —
~76 ms vs ~79 ms — because the offset *walk* itself dominates, not the heap
fetches. It earns its keep on very wide rows / cold cache. The real fix is
still keyset (§2); treat deferred join as mitigation, not cure.

## 4. `COUNT(*)` — "page 1 of N" is the second killer

`SELECT count(*) WHERE <filters>` scans everything the filter matches — often
slower than the page itself. Options:

1. **Don't show total pages** (Google-style "many more", infinite scroll).
2. **Estimate:** `EXPLAIN` row estimate or `pg_class.reltuples` for unfiltered
   counts (instant, ±10%).
3. **Cache the count** and refresh periodically / on write (counter table,
   materialized view — lab 07).
4. **Cap it:** `SELECT count(*) FROM (SELECT 1 ... LIMIT 10001)` → "10000+".

## 5. Cursors — for bulk export, not APIs

`DECLARE c CURSOR FOR ...; FETCH 1000 FROM c;` streams without re-scanning,
but holds a transaction + snapshot open (blocks `VACUUM` — lab 06). Use for
batch jobs; use keyset for request/response APIs.

## Exercises

1. Time `OFFSET 10` vs `OFFSET 400000` in the demo. Ratio? Now the keyset
   equivalents. Which stays flat?
2. Add a `WHERE status='active'` filter — what composite index does keyset
   need? (`(status, id)`.)
3. Run the `COUNT(*)` with and without filter. Which one hurts? Replace with
   the `reltuples` estimate — how close?
