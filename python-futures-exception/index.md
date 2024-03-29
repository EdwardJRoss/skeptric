---
categories:
- python
date: '2020-07-21T21:41:16+10:00'
image: /images/back-to-the-future-logo.svg
title: Raising Exceptions in Python Futures
---

Python [`concurrent.futures`](https://docs.python.org/3/library/concurrent.futures.html) are a handy way of dealing with asynchronous execution.
However if you're not careful it will swallow your exceptions leading to difficult to debug errors.

While you can perform [concurrent downloads with multiprocessing](/multiprocess-download) it means starting up multiple processes and sending data between them as pickles.
One problem with this is that you can't pickle some kinds of objects and often have to refactor your code to use multiprocessing.
It's also just unsatisfying having to spin up a bunch of processes when you're not really utilising them.

An alternative method is to use futures to perform the tasks asynchronously.
Since most of the time is spent waiting for file I/O this has a similar speed up:

```python
from concurrent import futures
with futures.ThreadPoolExecutor() as executor:
  results = futures.wait([executor.submit(download, filename) for filename in filenames])
```

However if there's some problem with one of the downloads we'll never actually see the error.
This is because asynchronous programming is hard; we may not want to fail if one file fails.
This gives us the choice of how to resolve it, but it means we have to be diligent to resolve it.

If we just want to raise the exception we can do it by evaluating the results:

```python
from concurrent import futures
with futures.ThreadPoolExecutor() as executor:
  results = futures.wait([executor.submit(download, filename) for filename in filenames])
for result in results.done:
  result.result()
```

This works because when we evaluate `.result()` it brings it out of the async world back into normal programming and raises the exception.
However this isn't at all obvious, so we can be a little more explicit:

```python
from concurrent import futures
with futures.ThreadPoolExecutor() as executor:
  results = futures.wait([download(filename) for filename in filenames])
for result in results.done:
  if result.exception() is not None:
    raise result.exception()
```

That works but it's a bit painful to remember every time.
I wonder if there are other asynchronous frameworks that make harder to overlook errors.
I've heard that Erlang/Elixir force you to specify error handling in every sent message; I'd be interested in how that works at scale.