---
categories:
- sql
- legacy code
- r
date: '2020-09-18T19:12:08+10:00'
image: /images/dbplyr_diff.png
title: Diffing in SQL
---

One way of refactoring legacy code is to use [diff tests](/diff-tests); checking what changes when you change the code.
While it can be easy to `diff` files, it's a little less obvious how to do this with SQL pipelines.
Fortunately there are a few different techniques to do this.

For exact matching you can use union all to find the number of rows that don't occur in both datasets.
For approximate matching you can use a join to check whether the differences are within some bounds.
These techniques work well together: for an approximate match you can first check the keys with union all, and then check the values with join.

# Exact Matching: Union All

A simple way of checking whether two tables are the same is to use `UNION ALL`.
This query will return all the rows that occur in only one or the other table, and which table ('old' or 'new') that they occur in.

```sql
SELECT col_1, col_2, ..., col_n,
       count(*) as mult,
       max(source) as source
FROM
(
  SELECT col_1, col_2, ..., col_n,
         'old' as source
  FROM A

  UNION ALL

  SELECT col_1, col_2, ..., col_n,
         'new' as source
  FROM B
)
GROUP BY col_1, col_2, ..., col_n
HAVING COUNT(DISTINCT source) <> 2
```

The only downside of this is it can be a lot of typing if the tables have a lot of columns to type in.
You could always template this in a programming language; for example in the excellent [`dbplyr`](https://github.com/tidyverse/dbplyr) you can easily do a variation like this:

```R
union_all(table_a %>% mutate(source='old'),
          table_b %>% mutate(source='new')) %>%
group_by(across(-c('source'))) %>%
filter(n_distinct(source) != 2)
```

# Approximate Matching: Join

Sometimes the results are allowed to vary a little bit because of slight changes to the data, random inputs and race conditions.
In this case the union all approach will return much more than we want.
The union all is still useful for checking both tables have the *same keys*; we can then use a join to check the values are similar.

For example to get all rows where the column `val` changes by more than 5% you could use something like:

```sql
SELECT *
FROM A
JOIN B on a.key_1 = b.key_1 and a.key_2 = b.key_2 ...
WHERE a.val NOT BETWEEN 0.95 * b.val and 1.05 * b.val
```

These kinds of approximate joins require a lot more thought; how much variation is expected?
Maybe it's ok for 1% of the rows to vary by more than 5%, but only 0.1% of the rows to vary by more than 10%.
However when you've decided the rules they're pretty straightforward to implement in SQL.
