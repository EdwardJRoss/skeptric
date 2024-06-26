---
categories:
- sql
date: '2020-08-27T08:00:00+10:00'
image: /images/left_join.png
title: Filtering a left join in SQL
---

When doing a left join in SQL any filtering of the table after the join will turn it into an inner join.
However there are some easy ways to do the filtering first.

Suppose you've got some tables related to a website.
The `pages` table describes the different pages on the site.

| page_id | page_name |
|---------|-----------|
| 1       | home      |
| 2       | checkout  |
| 3       | terms     |

Another `pageviews` describes the daily activity:

| date       | page_id | views |
|------------|---------|-------|
| 2020-08-27 | 1       | 100   |
| 2020-08-27 | 2       | 30    |
| 2020-08-28 | 1       | 150   |
| 2020-08-28 | 2       | 17    |
| 2020-08-28 | 3       | 2     |

We want to see the page views with the page name for all the pages on a certain date, even the ones with no views.
One wrong attempt would be:

```sql
select page_name, coalesce(views, 0) as views
from pages
left join pageviews on pages.page_id = pageviews.page_id
where date = '2020-08-27'
```

The problem is that the where clause will filter out the terms page from the results.

| page_name | views |
|-----------|-------|
| home      | 100   |
| checkout  | 30    |

This could be fixed by moving the where clause into a subquery:

```sql
select page_name, coalesce(views, 0) as views
from pages
left join (
  select *
  from pageviews
  where date = '2020-08-27'
) pv on pv.page_id = pages.page_id
```

This is verbose, but gives the correct result.

| page_name | views |
|-----------|-------|
| home      | 100   |
| checkout  | 30    |
| terms     | 0     |

A very useful trick is to move the filter condition into the join condition.

```sql
select page_name, coalesce(views, 0) as views
from pages
left join pageviews
  on pages.page_id = pageviews.page_id
  and date = '2020-08-27'
```

Because the `on` effectively filters before the output this gives the same result as the subquery:

| page_name | views |
|-----------|-------|
| home      | 100   |
| checkout  | 30    |
| terms     | 0     |

Adding filter conditions with `on` is useful for simplifying filters like this.
It's also sometimes clearer to put filter conditions closer to the table they apply to rather than at the end in a where clause.
