---
categories:
- python
date: '2021-04-17T15:31:26+10:00'
image: /images/sqlitedict.png
title: Persistent Dictionaries in Python
---

Dictionaries in Python (in other languages called maps or hashmaps) are a useful and flexible data structure that can be used to solve lots of problems.
Part of their charm is the affordances in the lanugage for them; setting and accessing with square brackets `[]`, deleting with `del`.
But sometimes you want a dictionary that persists across sessions, or can handle more data than you can fit into memory - and there's a solution persistent dictionaries.

Python has an inbuilt solution called [shelve](https://docs.python.org/3/library/shelve.html) which does this using [dbm](https://docs.python.org/3/library/dbm.html).
There's also the [sqlitedict](https://github.com/RaRe-Technologies/sqlitedict) library (from the makers of Gensim), which builds on sqlite.
They both allow strings as keys and arbitrary pickleable objects as values.
The best thing is they look and feel like ordinary dictionaries (with the caveat that if you mutate the value it's only updated on disk with an explicit assignment).
Sqlitedict has some advantages coming from sqlite, including handling concurrent access and multiple threads (although internally writes are serialised).

If you're running out of memory in a dictionary putting it into one of these offloads it to disk, which means you can free up a lot of memory at the cost of slower access and modification.
When running a periodic batch job, or a process that may fail, this gives a way to save state between runs.
You could even use it as a simple task queue between processes.
You can then back up the state simply by backing up the files (as long as they're not being written to).

There are many cases when you'll want a more robust solution, like persisting to a more versatile database like Postgres (or even using SQLite directly), or maintaining a proper task queue like Celery on Redis or RabbitMQ.
But all this infrastructure has a lot of overhead; for many cases it's just easier to use file-base databases through a dictionary interface.