---
categories:
- data
- sql
date: '2020-08-12T23:00:00+10:00'
image: /images/double_tick.svg
title: Data Tests with SQL
---

A challenge of data analytics is that the *data* can change as well as the code.
The systems producing and collecting data are often changed and can lead to missing or corrupt data.
These can easily corrupt reports and machine learning systems.
Worst of all the data may be lost permanently.
So if you're going to use some data it's important to check the data regularly to catch the worst kind of mistakes as early as possible.

The right tests will depend on your scenario, but there are a lot of common ones in SQL.
Checking that a column is non-null, or that the values are [unique](/unique-counts-sql) or that one column is contained in another.
Ideally these would be validated in the database itself, but often are not for implementation reasons, and sometimes it's good enough to be only true most of the time (say, 98% of the time).
Other examples are checking daily counts are within a certain range, checking the values a column can take, or checking that the values are withing some range (for example dates are not in the future). 

When writing tests the key is to keep them as simple and obviously correct as possible.
The easiest way to do this is to keep each test a *separate* query (which may seem inefficient, but is worth the savings in debugging time).
It's also useful to build up intermediate tables or [views](/sql-view) to keep the logic clearer and make debugging easier.

If you're really interested in making sure your report is correct make sure you do these checks in the same transaction with [enough isolation](https://www.postgresql.org/docs/9.5/transaction-iso.html); in particular that reads are repeatable.
Otherwise you may be testing different data to the data you use.