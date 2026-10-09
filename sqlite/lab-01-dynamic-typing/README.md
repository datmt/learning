# Lab 1 — Dynamic Typing, Type Affinity, STRICT Tables

**Question:** if I declare `n INTEGER`, can I store `'hello'` in it?

**Answer:** yes, by default. In SQLite the *value* has a type, not the *column*.
The column type is only a **hint** (called *affinity*). `STRICT` tables turn checking back on.

```bash
python3 demo.py
```

## 1. Storage classes vs affinity

Every value is one of 5 **storage classes**: `NULL`, `INTEGER`, `REAL`, `TEXT`, `BLOB`.
`typeof(x)` tells you which one was actually stored.

Every column has one of 5 **affinities**: `INTEGER`, `REAL`, `NUMERIC`, `TEXT`, `BLOB`.
Affinity means "*try* to convert to this type when storing. If you can't without losing data, keep the original."

| Inserted into `n INTEGER` | Stored as |
|---|---|
| `'42'` | `42` integer (converted) |
| `'1e3'` | `1000` integer (converted) |
| `'hello'` | `'hello'` text (kept, no error!) |

## 2. How affinity is chosen from the type name (substring rules, in order)

1. contains `INT` → INTEGER
2. contains `CHAR`, `CLOB`, `TEXT` → TEXT
3. contains `BLOB` or no type → BLOB (no conversion)
4. contains `REAL`, `FLOA`, `DOUB` → REAL
5. otherwise → NUMERIC

Consequences you will see in the demo:

- `VARCHAR(3)` is **TEXT** and the `(3)` is ignored. A 18-char string is stored fine.
- `FLOATING POINT` contains `INT` (in "POINT") → **INTEGER** affinity. `'12'` becomes `12`.
- `STRING` matches nothing → **NUMERIC**. `'12'` becomes integer `12`. Use `TEXT`, not `STRING`.
- `BANANA` is accepted. Any name is legal.
- `DATETIME` → NUMERIC; `'2024-01-01'` cannot convert, so it stays text.

## 3. Comparison quirks

Sort order across classes: `NULL < INTEGER/REAL < TEXT < BLOB`.

- `SELECT '9' > 10` → `1` (text is always "bigger" than numbers).
- A column with no type stores `'9'` as text, so `WHERE x = 9` finds **nothing**.
  This is the classic bug when an app sometimes binds `"9"` and sometimes `9`.

## 4. No boolean, no date type

- `TRUE` / `FALSE` are integers `1` / `0`.
- Dates are stored as TEXT (ISO-8601), REAL (Julian day) or INTEGER (unix epoch). You choose.
  Date functions (`date()`, `datetime()`, `unixepoch()`, `strftime()`) work on all three.
- Quirk: `date('2024-01-31', '+1 month')` → `'2024-03-02'` (Feb 31 overflows into March).

## 5. Numbers

- `7 / 2` = `3` (integer division). Write `7 / 2.0` or `CAST(x AS REAL)`.
- `9223372036854775807 + 1` silently becomes a REAL (precision lost!).
- `sum()` is stricter and raises `integer overflow`. (`total()` always returns REAL.)

## 6. STRICT tables (SQLite 3.37+, 2021)

```sql
CREATE TABLE t (id INTEGER PRIMARY KEY, n INTEGER, s TEXT, any_col ANY) STRICT;
```

- Allowed types only: `INT, INTEGER, REAL, TEXT, BLOB, ANY`. `DATETIME` → error.
- Lossless conversion still happens (`'42'` → `42`), but `'hello'` → `cannot store TEXT value in INTEGER column`.
- `ANY` stores exactly what you give, no conversion.

## Production tips

- **Use `STRICT` for new tables.** Type bugs show up at insert time, not months later.
- If you can't, add `CHECK (typeof(n) = 'integer')`.
- Bind parameters with consistent Python types. `"9"` and `9` are different values.
- Pick one date format per column (recommend unix epoch INTEGER or ISO-8601 TEXT in UTC) and stick to it.

## Exercise

1. Create `CREATE TABLE p (price NUMERIC)`. Insert `'1.50'`, `'1.5'`, `'abc'`, `'0x10'`. Predict `typeof()` for each, then check.
2. Add a `CHECK` that makes a non-STRICT column reject text.
