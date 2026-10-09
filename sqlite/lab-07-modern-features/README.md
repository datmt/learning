# Lab 7 — Modern Features (JSON, Full-Text Search, Window Functions, ...)

**Question:** is SQLite a "toy" SQL engine?

**Answer:** no. Modern SQLite has JSON, full-text search, window functions, recursive CTEs,
UPSERT, RETURNING, generated columns and FULL OUTER JOIN. Many apps don't need a separate
search engine or document store.

```bash
python3 demo.py
```

Some features depend on how SQLite was compiled (FTS5, math functions). Python's bundled
SQLite usually has them. Check yours: `SELECT * FROM pragma_compile_options;`

## 1. JSON (built in since 3.38)

| Syntax | Returns |
|---|---|
| `body -> '$.a.b'` | JSON text (`'["a","b"]'`) |
| `body ->> '$.a.b'` | SQL value (`'ann'`, `120`) |
| `json_each(body, '$.tags')` | one row per array element (table-valued function) |
| `json_object(...)`, `json_group_array(...)` | build JSON from rows |
| `json_set / json_insert / json_replace / json_remove` | modify |
| `jsonb(...)` (3.45+) | binary JSON blob, avoids re-parsing text |

JSON is stored as plain TEXT. **Index a JSON path** with an expression index or a generated column:

```sql
CREATE INDEX ix_tier ON ev (body ->> '$.user.tier');               -- plan: SEARCH ... (<expr>=?)
ALTER TABLE ev ADD COLUMN tier TEXT GENERATED ALWAYS AS (body ->> '$.user.tier') VIRTUAL;
```

## 2. FTS5 full-text search

```sql
CREATE VIRTUAL TABLE docs USING fts5(title, body, tokenize='porter unicode61');
SELECT title, snippet(docs, 1, '[', ']', '...', 6) FROM docs WHERE docs MATCH 'reader*' ORDER BY rank;
```

- `porter` = English stemming (`writers` matches `writer`). `unicode61` = Unicode-aware tokenizing.
- Query syntax: `AND OR NOT`, `"phrase"`, `prefix*`, `column: term`, `NEAR(a b, 5)`.
- `ORDER BY rank` = BM25 relevance. `snippet()` / `highlight()` for UI.
- This replaces `LIKE '%word%'` (always a full scan, lab 6).
- Keep it in sync with your main table using an "external content" table + triggers (see SQLite FTS5 docs, section 4.4.3).

## 3. Window functions (3.25+)

`sum() OVER (PARTITION BY ... ORDER BY ...)`, `rank()`, `row_number()`, `lag()`, `lead()`,
`first_value()`, `ntile()`. Running totals and top-N-per-group without self-joins.

## 4. Recursive CTE

Generate series, walk trees/graphs (org charts, folders, comment threads).

## 5. UPSERT + RETURNING

```sql
INSERT INTO counter VALUES ('home', 1) ON CONFLICT(key) DO UPDATE SET hits = hits + 1;
INSERT INTO ev (body) VALUES ('{}') RETURNING id;
```

`INSERT OR REPLACE` is **not** an upsert: it *deletes* the old row (fires delete triggers,
cascades FKs, assigns a new rowid if not given). Prefer `ON CONFLICT DO UPDATE`.

## 6. Misc

- Math: `sqrt, pow, ln, exp, ...` (needs `SQLITE_ENABLE_MATH_FUNCTIONS`; on in most builds).
- `string_agg` (3.44), `iif`, `format`/`printf`, `FILTER (WHERE ...)` on aggregates.
- `RIGHT` and `FULL OUTER JOIN` (3.39+).

## What SQLite does NOT have

- No stored procedures, no user accounts/GRANT, no `ALTER COLUMN TYPE`.
- No built-in network server (it's a library).
- Custom functions: register them from your app (`conn.create_function` in Python).

## Exercise

1. Add a `NEAR` query to the FTS5 demo.
2. Using a window function, find the best day per region (`row_number() ... = 1`).
3. Register a Python function `reverse(s)` with `db.create_function` and use it in SQL.
