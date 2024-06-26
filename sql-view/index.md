---
categories:
- sql
- data
date: '2020-04-21T22:18:46+10:00'
image: /images/create_view.png
title: SQL Views for hiding business logic
---

The longer I work with a database the more I learn the dark corners of the dataset.
Make sure you exclude the rows created by the test accounts listed in another table.
Don't use the `create_date` field, use the `real_create_date_v2` instead, unless it's not there, then just use `create_date`.
Make sure you only get data from the latest snapshot for the key.

Very quickly I end up with complex spaghetti SQL, which either contains monstrous subqueries or a chain of `CREATE TEMPORARY TABLE`.
Moreover everyone I talk to has a slightly different logic for working around these issues, and even I'm not consistent between analyses.
This a perfect usecase for SQL Views.

A view is just a way of treating the result of a query as a table.
So once I've got my complex query that excludes the right rows, has sensible column names and deduplicates the rows I can wrap it in a `CREATE VIEW` statement and then reuse it later.
Moreover when I find out a new issue in the dataset I can update my `CREATE VIEW` statement and my previous reports will be automatically corrected; which is great as long as you can manage expectations about reporting consistency.
This is much nicer than creating a table as it will always be up to date with the underlying dataset.

This is just a small part of the solution; documenting the datasets and having appropriate monitoring are also really important.
But having convenient views helps work around footguns in the underlying datasets that can't be changed, meaning you're more likely to get the right data and get it faster.

One issue that can arise is dependencies.
If many other people start using it you may get to a point where you can't make changes because it will impact other people.
There's lots of different solutions, but the easiest may be to *version* your view (maybe putting it in an archive schema if people still need the historical version).

Another issue is performance.
Whenever you query the view you're rerunning the view's query, which can be slow if it's doing a lot of work.
One great solution (if your database supports it) is a [`MATERIALIZED VIEW`](https://en.wikipedia.org/wiki/Materialized_view).
This caches the result of the underlying query to make evaluating it much faster.
Otherwise you may need to switch the view to a table and setup a process that rebuilds it regularly.
You need to be careful though because if your process breaks you may end up with stale data (and not even know about it).

One interesting application of SQL Views is to expose a table in a different way to how the underlying data is stored.
This kind of approach is encouraged in [PostgREST](http://postgrest.org) which exposes a HTTP API from PostgreSQL tables and views.
When you need to change the underlying data structure you can abstract these changes away from the API with a view.

SQL Views are a useful way of hiding some complexity of queries that you often reuse.
Like any shared asset you have to think about dependencies on it as you change it, but it offers a way to reuse logic accross different queries with minimal maintenance (as opposed to a table you have to keep up to date).