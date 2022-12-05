---
categories:
- sql
date: '2020-07-14T20:09:30+10:00'
image: /images/cross_join.svg
title: Filling Gaps in SQL
---

It's common for there to be gaps or missing values in an SQL table.
For example you may have daily traffic by source, but on some low volume days around Christmas there are no values in the low traffic sources.
Missing values can really complicate some calculations like [moving averages](/moving-averages-sql), and some times you need a way of filling them in.
This is straightforward with a cross join.

You need all the possible variables you're filling in, and the value to fill.
For example you might have the `daily` table containing each device, traffic source and date the number of visitors.

| device  | source | date       | visitors |
|---------|--------|------------|----------|
| desktop | direct | 2019-12-23 | 100      |
| desktop | paid   | 2019-12-23 | 3        |
| mobile  | any    | 2019-12-23 | 32       |
| desktop | direct | 2019-12-24 | 40       |
| mobile  | any    | 2019-12-24 | 18       |
| desktop | direct | 2019-12-26 | 80       |
| desktop | paid   | 2019-12-26 | 2        |
| mobile  | any    | 2019-12-26 | 23       |

There are some missing combinations of valid (device, source) and date with visitors, and we want to fill them with 0s.
We could get all the possible combinations of device and source with a simple query:

```sql
select device, source
from daily
group by 1, 2
```

Similarly we could generate all possible dates; if some may be missing for every device and source it's best to use something like [generate series](https://www.postgresql.org/docs/9.1/functions-srf.html#FUNCTIONS-SRF-SERIES):

```sql
SELECT date
FROM generate_series(
  (select min(date) from daily),
  (select max(date) from daily),
  '1 day') as dates(date)
```

Then we can generate all possible combinations with a cross join.
The exact syntax may vary by database; I'll assume here that separating with commas does the trick (at least Presto and Postgres).

```sql
select device, source, date
from (
  select device, source
  from daily
  group by 1, 2
), 
 generate_series(
  (select min(date) from daily),
  (select max(date) from daily),
  '1 day') as dates(date)
```

Finally we can fill in the data from the underlying table with a coalesce, using the default value of 0 for missing entries:

```sql
select device, source, date, 
       coalesce(daily.users, 0) as users
from (
  select device, source, date
  from (
    select device, source
    from daily
    group by 1, 2
  ), 
   generate_series(
    (select min(date) from daily),
    (select max(date) from daily),
    '1 day') as dates(date)
) xjoin
left join daily
  on xjoin.device = daily.device 
  and xjoin.source = daily.source
  and xjoin.date = daily.date
```

Finally it's always good to verify the table is actually as expected.
Suppose we materialised the above query into `daily_full`:

```sql
select device, source,
       min(date) as first_date, max(date) as last_date, 
       count(distinct date) as dates,
       count(*) as rows,
       count(users) as user_rows
```

We would expect the last 3 columns to be the same and equal to the number of days between first date and last date (which you could add with the database specific function to calculate the difference between dates).

This is a useful pattern for filling in missing values in tables, and can be used in more complex scenarios where the filled value depends on the parameters.