---
categories:
- presto
- athena
- sql
date: '2020-03-26T08:00:00+11:00'
image: /images/sql_window_function.png
title: Getting most recent value in Presto with max_by
---

[Presto](https://prestodb.io) and the AWS managed alternative [Amazon Athena](https://aws.amazon.com/athena/) have some powerful [aggregation functions](https://prestodb.io/docs/current/functions/aggregate.html) that can make writing SQL much easier.
A common problem is getting the most recent status of a transaction log.
The `max_by` function (and its partner `min_by`) makes this a breeze.

Suppose you have a table tracking user login activity over time like this:

| country | user_id | time | status |
| ------- | ------- | ---- | ------ |
|   AU    |    1    | 2020-01-01 08:00 | logged-in |
|   CN    |    2    | 2020-01-01 09:00 | logged-in |
|   AU    |    1    | 2020-01-01 12:00 | logged-out |
|   AU    |    1    | 2020-01-01 13:00 | logged-in |
|   CN    |    2    | 2020-01-01 14:00 | logged-out |

You need to find out which users are currently logged in and out, which requires you to find their most recent status.
In standard SQL you can do this with a window function by adding a row_number to find the most recent time:

```SQL
SELECT user_id, status
FROM (
    SELECT
      user_id,
      status,
      row_number() OVER (PARTITION BY user_id ORDER BY time DESC) AS rn
    FROM user_activity
)
WHERE rn = 1
```

With Presto's `max_by` function you can do this in a single query:

```SQL
SELECT user_id, max_by(status, time) AS status
FROM user_activity
GROUP BY user_id
```

There is one downside to this approach: if you also try to select another column like `max_by(country, time)` there's a chance if there are two rows with the same time we will get the most recent `status` and `country` from different rows which could have consistency problems.

An extension is you may want to also get their *previous* status.
In standard SQL you could use a window function:

```SQL
SELECT user_id, status, last_status
FROM (
    SELECT
      user_id,
      status,
      lag(status) OVER (PARTITION BY user_id ORDER BY time DESC) AS last_status,
      row_number() OVER (PARTITION BY user_id ORDER BY time DESC) AS rn
FROM user_activity
)
WHERE rn = 1
```

In Presto you can pass an additional arugment to `max_by` on how many values to return in an array.

```SQL
SELECT user_id, max_by(status, time, 2) AS last_2_statuses_array
FROM user_activity
GROUP BY user_id
```

One more trick Presto has is `count_if` which removes case statements from aggregation.
For example if we wanted the number of logged in and logged out users by country in a pivoted view for standard SQL we could write

```SQL
SELECT
  country,
  count(CASE WHEN status = 'logged-in' THEN 1 end) AS logged_in_users,
  count(CASE WHEN status = 'logged-out' THEN 1 end) AS logged_out_users
FROM (
    SELECT
      user_id,
      country,
      status,
      row_number() OVER (PARTITION BY user_id ORDER BY time DESC) AS rn
    FROM user_activity
)
WHERE rn = 1
GROUP BY country
ORDER BY count(CASE WHEN status = 'logged-in' THEN 1 end) DESC
```

But in Presto with `count_if` it could be:

```SQL
SELECT country,
       count_if(status = 'logged-in') AS logged_in_users,
       count_if(status = 'logged-out') AS logged_out_users
FROM (
    SELECT
      user_id,
      max_by(country, time) AS country,
      max_by(status, time) AS status
    FROM user_activity
    GROUP BY user_id
)
GROUP BY 1
ORDER BY 2 DESC
```

When you're writing a lot of complex queries even small simplifications add up.
It would be great to see these kinds of functions in other databases some day.
