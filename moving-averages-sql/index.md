---
categories:
- sql
date: '2020-03-27T08:00:43+11:00'
image: /images/moving_average.png
title: Moving Averages in SQL
---

Moving averages can help smooth out the noise to reveal the underlying signal in a dataset.
As they lag behind the actual signal they tradeoff timeliness for increased precision in the underlying signal.
You could use them for reporting metrics or for alerting in cases where it's more important to be sure there is a change than it is to catch any change early.
It's typically better to have a 7 day moving average than weekly reporting for important metrics because you'll see changes earlier.
There are a few ways to implement this in SQL with different tradeoffs, and a few traps to avoid.

The simplest way is with by summing over a limited window, but you have to be careful about missing data.
It's possible to construct a window manually with multiple lags which can let you choose weights.
Or finally you can use a self join which can handle missing data and flexible weighting.
Depending on your situation and database it's worth considering which one is best in terms of performance and simplicity.

My recommendation in general is to use a self-join with a weights table ([fiddle](https://dbfiddle.uk/?rdbms=postgres_11&fiddle=c2db62742ff071c49c0ea6d7b8808f94)):

```SQL
SELECT pages.date,
       max(pages.pageviews) as pageviews,
       CASE
       WHEN pages.date - (select min(date) from pages) >= 2
       THEN sum(weight * ma_pages.pageviews)
       END as weighted_moving_average
FROM pages
JOIN pages AS ma_pages ON pages.date - ma_pages.date BETWEEN 0 AND 2
JOIN weights ON idx = pages.date - ma_pages.date
GROUP BY pages.date
ORDER BY pages.date
```

The rest of the article will go through the options and how to work around comming issues with missing data and handling the first few rows properly.

# Calculating a moving average

Suppose you have the daily number of pageviews for a new website you're developing.
To remove some of the noise you want to calculate a 3 day moving average (although in real life 7 day would be better because it smooths out weekend effects).
Here is an example of the output:

| date       | pageviews | moving_average |
| ---------- | --------- | -------------- |
| 2020-02-01 | 42        | 42             |
| 2020-02-02 | 3         | 22.5           |
| 2020-02-03 | 216       | 87             |
| 2020-02-04 | 186       | 135            |
| 2020-02-05 | 510       | 304            |
| 2020-02-06 | 419       | 371.667        |
| 2020-02-07 | 64        | 331            |
| 2020-02-09 | 230       | 98             |

Look particularly at the first two rows where there's not a full 3 day window and the last row where it comes after a missing date.

In practice you might be calculating this for different segments, or for different periods (like weekly/monthly/quarterly), but the overall approach will be the same.

# Moving window frame

The easiest way is with a moving window frame; you might start with something like:

```SQL
-- Don't do this if there might be missing dates
SELECT *,
      avg(pageviews) OVER (
        ORDER BY date
        ROWS BETWEEN
          2 PRECEDING AND
          CURRENT ROW
      ) AS moving_average
FROM pages
ORDER BY date
```

Note that for an N day moving window you use `BETWEEN N-1 PRECEDING` in the frame clause.

However there's a problem here: if you've got missing days then it's going to grab extra data before the moving window.
For example in our table above there's no data for 2020-02-08 so the query above will get data from 2020-02-06 which is more than 3 days ago.

There's a build in way to solve this using the `RANGE` clause rather than the `ROWS` clause.
In databases that support this with dates, like [PostgreSQL 11](https://www.postgresql.org/docs/11/sql-expressions.html#SYNTAX-WINDOW-FUNCTIONS), it's easy to fix ([fiddle](https://dbfiddle.uk/?rdbms=postgres_11&fiddle=5605072830420611fd222920f5123ed4)).

```SQL
SELECT *,
      avg(pageviews) OVER (
        ORDER BY date
        RANGE BETWEEN
          '2 DAYS' PRECEDING AND
          CURRENT ROW
      ) AS moving_average
FROM pages
ORDER BY date
```

However not many databases support this, but some others support integer ranges.
You could use the relevant date functions to create a date offset index:

```SQL
SELECT *,
      avg(pageviews) OVER (
        ORDER BY date_offset
        RANGE BETWEEN
          2 PRECEDING AND
          CURRENT ROW
      ) AS moving_average
FROM (
    SELECT *,
           date - min(date) AS date_offset
    FROM pages
) as pages_offset
ORDER BY date
```

However support for bounded `RANGE` is pretty weak in databases, so sometimes not even this is an option.
Another drawback of the `RANGE` solution is we don't have the moving average value for 2020-02-08, even though it will have a value.
The remaining solution is to fill out the table so there's a row for each date with 0 page views.
The general strategy is to create another table that has every date between the maximum and minimum of the `pages` and coalesce pageviews with 0; how you do this is database dependent.
Then you join this to the `pages` table and fill in the nulls with a `coalesce` ([fiddle](https://dbfiddle.uk/?rdbms=postgres_11&fiddle=7d2c81513dd77d9d6dacc4e9b816602c)).

```SQL
-- PostgresSQL example
SELECT *,
       avg(pageviews) OVER (
         ORDER BY date
         ROWS BETWEEN
           2 PRECEDING AND
           CURRENT ROW
       ) AS moving_average
-- Generating a date table is database dependent
FROM (
SELECT dates.date, coalesce(pageviews, 0) AS pageviews
FROM generate_series((select min(date) from pages),
                      (select max(date) from pages),
                      '1 day') as dates
LEFT JOIN pages on dates.date = pages.date
) AS pages_full
ORDER BY date
```

Note that if we were calculating the pageviews by segment we could just update the window function to be `OVER (PARTITON BY SEGMENT ORDER BY ...)`.

There's a limitation with this approach; it's not possible to do a weighted moving average.

# Moving Averages with Lag

Another way to do moving averages is by selecting the previous rows with the lag window function.
This tends to be very verbose, but a benefit is you can choose weights for each point.
A weighted moving average is useful because you can weight down further ago values to capture more of the trend, so the moving average does not lag the signal as much.

The solution with lag is straightforward, but [tedious](https://dbfiddle.uk/?rdbms=postgres_11&fiddle=6dd6f9102117afd1143161aa1f6a45fe) (especially if you need to make a 90 day moving window):

```SQL
-- Don't use with missing dates
SELECT *,
      (pageviews +
       LAG(pageviews) OVER (order by date) +
       LAG(pageviews, 2) OVER (order by DATE)) / 3 AS moving_average
FROM pages
ORDER BY date
```

The first two rows are null rather than the relevant average - depending on your application this may be more or less appropriate.
More problematically if there are missing dates then it will get the wrong result like our first `ROWS` query.
It's possible to work around this by throwing away data outside the date window ([fiddle](https://dbfiddle.uk/?rdbms=postgres_11&fiddle=14a545b36a79b566a9bb3824a19165e0)):

```SQL
SELECT *,
      (pageviews +
       (CASE
        WHEN (date - LAG(date) OVER (order by date)) <= 2
        THEN 1
        ELSE 0 END
       ) * LAG(pageviews) OVER (order by date) +
       (CASE
        WHEN (date - LAG(date, 2) OVER (order by date)) <= 2
        THEN 1
        ELSE 0 END
       ) * LAG(pageviews, 2) OVER (order by DATE)) / 3 AS moving_average
FROM pages
ORDER BY date
```

However as in the previous section the best solution is probably to join it with a full date table if there may be missing dates ([fiddle](https://dbfiddle.uk/?rdbms=postgres_11&fiddle=777ac73c01dcedf508b53f456f1dd300)):

```SQL
SELECT *,
      (pageviews +
       LAG(pageviews) OVER (order by date) +
       LAG(pageviews, 2) OVER (order by DATE)) / 3 AS moving_average
FROM (
SELECT dates.date, coalesce(pageviews, 0) AS pageviews
FROM generate_series((select min(date) from pages),
                      (select max(date) from pages),
                      '1 day') as dates
LEFT JOIN pages on dates.date = pages.date
) AS pages_full
ORDER BY date
```

## Adding weights

Because we're manually writing each part of the moving average it's possible to add weights; say we wanted to use the weights (0.6, 0.24, 0.16) to emphasise the more recent data points.
It's as simple as inserting the weights into the query:

```SQL
SELECT *,
      0.6 * pageviews +
      0.24 * LAG(pageviews) OVER (order by date) +
      0.16 * LAG(pageviews, 2) OVER (order by DATE) AS weighted_moving_average
FROM (
SELECT dates.date, coalesce(pageviews, 0) AS pageviews
FROM generate_series((select min(date) from pages),
                      (select max(date) from pages),
                      '1 day') as dates
LEFT JOIN pages on dates.date = pages.date
) AS pages_full
ORDER BY date
```

The lag approach is simple and should work in any database that supports window functions.
As before we can do it per segment using `PARTITION BY` in the window clause.
However writing each lag is tedious for large windows, which the next approach solves.

# Moving Averages with Self Joins

Using self joins is in some senses the simplest, most reliable and versatile.
Not every SQL database supports window functions, but they should support JOIN.
However you may opt for one of the other options for performance reasons, or for convenience in a quick analysis.

The basic approach is to join the table to itself over the range of dates; this looks really similar to the `RANGE` window ([fiddle](https://dbfiddle.uk/?rdbms=postgres_11&fiddle=098e3726d2503941b94b288332a65d85))

```SQL
-- Only use if doesn't have missing data
SELECT pages.date,
      max(pages.pageviews) as pageviews,
      avg(ma_pages.pageviews) as moving_average
FROM pages
JOIN pages AS ma_pages ON
       pages.date - ma_pages.date BETWEEN 0 AND 2
GROUP BY pages.date
ORDER BY pages.date
```

However again missing dates make the result incorrect.
In our example for 2020-02-09 the denominator for the average is 2 (because there's no row for 2020-02-10).
As before we can fix this by inserting 0 pageviews for the missing days ([fiddle](Fiddle: https://dbfiddle.uk/?rdbms=postgres_11&fiddle=b4ee2028eba4d6adbd7a2518c02d2bc3)).

```SQL
SELECT pages.date,
      max(pages.pageviews) as pageviews,
      avg(ma_pages.pageviews) as moving_average
FROM (
SELECT dates.date, coalesce(pageviews, 0) AS pageviews
FROM generate_series((select min(date) from pages),
                      (select max(date) from pages),
                      '1 day') as dates
LEFT JOIN pages on dates.date = pages.date
) AS pages
LEFT JOIN (
SELECT dates.date, coalesce(pageviews, 0) AS pageviews
FROM generate_series((select min(date) from pages),
                      (select max(date) from pages),
                      '1 day') as dates
LEFT JOIN pages on dates.date = pages.date
) AS ma_pages ON pages.date - ma_pages.date BETWEEN 0 AND 2
GROUP BY pages.date
ORDER BY pages.date
```

However there's another way we can fix this by using weights.

## Weighted moving average

To calculate the weighted moving average we can store the weights in a separate table.
For example if we want the most recent data point to have a weight of 0.6, the middle point a weight of 0.24 and the furthest point a weight of 0.16 we could have a table like this:

| idx | weight |
| --- | -----  |
| 0   | 0.6    |
| 1   | 0.24   |
| 2   | 0.16   |

Not that we could reproduce the moving average by having a table with each weight being equal and adding to 1

| idx | weight |
| --- | -----  |
| 0   | 0.333  |
| 1   | 0.333  |
| 2   | 0.333  |

We then join the weight based on the number of steps from the current date and calculate the inner product.
Note that we censor the first two rows with a CASE statement, otherwise they will be wrong ([fiddle](https://dbfiddle.uk/?rdbms=postgres_11&fiddle=9f0f1adcfac53da4829be234b88ca3b4)).

```SQL
SELECT pages.date,
       max(pages.pageviews) as pageviews,
       CASE
       WHEN pages.date - (select min(date) from pages) >= 2
       THEN sum(weight * ma_pages.pageviews)
       END as weighted_moving_average
FROM pages
JOIN pages AS ma_pages ON pages.date - ma_pages.date BETWEEN 0 AND 2
JOIN weights ON idx = pages.date - ma_pages.date
GROUP BY pages.date
ORDER BY pages.date
```

The best part about this is it works even if there's a missing date.
However you do lose the data point for the missing date, so you may want to complete the table if you know there's missing dates.
If we wanted to have partial results for the first 2 days we'd need to renormalise the weights based on the number of days since the first.
The only limitation to this method is you'll need a way to create the weights table.
But even if you don't have access to creating (temporary) tables, you may be able to do this using a [select from values](https://www.postgresql.org/docs/12/queries-values.html) ([fiddle](https://dbfiddle.uk/?rdbms=postgres_11&fiddle=55f87d80bd3244904519841e5dae69b7)).

```SQL
SELECT pages.date,
       max(pages.pageviews) as pageviews,
       CASE
       WHEN pages.date - (select min(date) from pages) >= 2
       THEN sum(weight * ma_pages.pageviews)
       END as weighted_moving_average
FROM pages
JOIN pages AS ma_pages ON pages.date - ma_pages.date BETWEEN 0 AND 2
JOIN (SELECT idx, 1/(2 + 1.) as weight FROM (VALUES (0, 1, 2)) as t(idx)) weights ON
  idx = pages.date - ma_pages.date
GROUP BY pages.date
ORDER BY pages.date
```

Now you know a few ways to create moving averages and how to avoid the most common pitfalls regarding missing data and the initial rows.
The weight table is the safest and most flexible solution and you could even create standard weight tables to use accross multiple metrics.
However sometimes you'll want to use the framed window method for performance or convenience methods (or the lag method if you also need weighting).

Happy querying!
