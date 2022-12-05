---
categories:
- SQL
- Presto
- Athena
date: '2020-03-28T08:24:51+11:00'
image: /images/athena_one_sixth.png
title: Calculating percentages in Presto
---

One trick I use all the time is calculating percentages in SQL by dividing with the count.
Percentages quickly tell me how much coverage I've got when looking at the top few rows.
However Presto uses integer division so doing the naive thing will always give you 0 or 1.
There's a simple trick to work around this: replace `count(*)` with `sum(1e0)`.

Suppose for example you want to calculate the percentage of a column that is not null; you might try something like

```SQL
SELECT count(col) / count(*) AS col_pct_nonnull
FROM some_table
```

However I was surprised when I got a 0; was the whole column null?
It is because Presto uses integer division by default.
So `1/6` gives 0 and `7/6` gives 1.

One way to work around this is by explicit casting to double:

```SQL
SELECT count(col) / cast(count(*) AS DOUBLE) AS col_pct_nonnull
FROM some_table
```

But that's a lot of typing very quickly.
We can try to coerce the number by adding a decimal, but this changes it to fixed precision in Presto.
So `SELECT 1/6., 1/6.0, 1/6.00` returns `0, 0.2, 0.17`.
This can be useful if you want to show a truncated percentage, e.g for 3 decimal points.
However in Athena these all coerce to floating point returning 0.1666...

```SQL
-- 3 decimal places in Presto
-- Floating point in Athena
SELECT count(col) / sum(1.000) AS col_pct_nonnull
FROM some_table
```

However I tend to prefer storing things in double precision; otherwise if you do something like calculate a cumulative sum the rounding errors can compound.
You can force this in Presto by explicitly using scientific notation; `SELECT 1/6e0` gives the result to double precision 0.16666...
So we could change our query to:

```SQL
SELECT count(col) / sum(1e0) AS col_pct_nonnull
FROM some_table
```

If you're ever doing a division in Presto or Athena it's good practice to throw in a `1e0 *` to make sure you're doing floating point arithmetic, otherwise you will often get misleading results.