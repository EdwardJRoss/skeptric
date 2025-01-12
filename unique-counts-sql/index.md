---
categories:
- sql
date: '2020-06-18T20:07:57+10:00'
image: /images/duplicate_rows.png
title: Checking for Uniques in SQL
---

When [checking my work](/checking) in SQL one of the first things I do is confirm a column I expect to be unique is.
Many tables have a unique key at the level they are at; for session level data it's a session id, for user level data it's a user_id or for daily data it's a date.
It's generally a useful thing to check because all it takes is one bad join to end up with a bunch of duplicate (or dropped) rows.
While this seems like an easy task if you're familiar with SQL it can quickly find it gets quite involved.

The easiest way is to compare the count of a column with the distinct count.
They will be the same only if the count is unique.

```sql
select count(user_id) as num_user,
       count(distinct user_id) as num_distinct_user
from users
```

It's generally good practice to check if there are any nulls in the column as well.
Checking whether two numbers is the same requires a little mental effort, so I get the computer to do the calculation and just check if the number is 0.

```sql
select count(user_id) as num_user,
       count(*) - count(user_id) as num_missing_user
       count(user_id) - count(distinct user_id) as num_duplicate_user
from users
```

If I'm ok with a small proportion number of rows missing or being repeated it can be more meaningful to report them as a ratio of the number of rows rather than an absolute number.
If there are no missing/duplicate rows then the last two columns will be 1.
I have taken to using `1e0` in divions because [Presto rounds integer division](/presto-integer-division), in some databases you could simply use `count(*)` in the denominator.

```sql
select count(*) as n, 
       count(user_id)/sum(1e0) as prop_not_missing,
       count(distinct user_id)/sum(1e0) as prop_distinct
from users
```

Another useful strategy is to find some examples of duplicate rows; they can be useful for diagnosing what's going wrong.
If the rows are distinct then this query will return no results, otherwise it will give the most frequently repeated rows.

```sql
select user_id, count(*) as n
from users
group by user_id
having count(*) > 1
order by n desc
limit 10
```

One benefit of this approach is it can be extended to multiple columns.
For example we might have a table that summarises user activity by day that is keyed on `user_id` and `action_date`.

```sql
select user_id, action_date, count(*) as n
from daily_users
group by user_id, action_date
having count(*) > 1
order by n desc
limit 10
```

We can build this into a single row summary as before by putting this into a subquery.

```sql
select sum(n) as num_user_days,
       sum(n) - count(*) as num_duplicate_user_days,
       sum(case
             when user_id is null 
               or action_date is null
             then n
           end) as num_missing_user_days
from (
    select user_id, action_date, count(*) as n
    from daily_users
    group by user_id, action_date
)
```

It's easy to create ratios as before.

```sql
select sum(n) as num_user_days,
       sum(n) / sum(1e0) as prop_unique_user_days,
       sum(case
             when user_id is not null 
               and action_date is not null
             then n
           end)  / sum(n*1e0) as prop_not_missing_user_days
from (
    select user_id, action_date, count(*) as n
    from daily_users
    group by user_id, action_date
)
```

You can even combine this to also get the most frequent duplicate in a single query for diagnosis.

```sql
select sum(n) as num_user_days,
       sum(n) / sum(1e0) as prop_unique_user_days,
       sum(case
             when user_id is not null 
               and action_date is not null
             then n
           end)  / sum(n*1e0) as prop_not_missing_user_days
       max(case when rn = 1 then user_id end) as most_common_pair_user,
       max(case when rn = 1 then action_date end) as most_common_pair_action,
       max(n) as n_most_common_pair,
       max(n)/sum(1e0) as prop_most_common_pair
from (
    select user_id, action_date, n,
    from (
           row_number() over (order by n desc) as rn
        from (
            select user_id, action_date, count(*) as n
            from daily_users
            group by user_id, action_date
        )
    )
)
```

By this stage it becomes moderately complex to maintain.
You could put it as an SQL template in your programming language of choice, or build it as a [dbplyr](https://dbplyr.tidyverse.org/) function.