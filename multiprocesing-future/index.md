---
categories:
- python
date: '2020-09-13T08:09:00+10:00'
image: /images/concurrent_futures.png
title: From Multiprocesing to Concurrent Futures in Python
---

Waiting for independent I/O can be a performance bottleneck.
This can be things like downloading files, making API calls or running SQL queries.
I've already talked about how to [speed this up with multiprocessing](/multiprocess-download).
However it's easy to move to the more recent [concurrent.futures](https://docs.python.org/3/library/concurrent.futures.html) library which allows running on threads as well as processes, and allows handling more complicated asynchronous flows.

From the previous post suppose we have this multiprocessing code:

```python
from multiprocessing import Pool
from urllib.request import urlretrieve

url_dests = [('http://example.com', 'example.html'), ...]
with Pool(8) as p:
    p.starmap(urlretrieve, url_dests)
```

Turning this into futures code is very easy; we replace a multiprocessing pool with a futures PoolExecutor, and `starmap(f, xs)` with `map(f, *zip(*xs))`.

```python
from concurrent.futures import ProcessPoolExecutor
from urllib.request import urlretrieve

url_dests = [('http://example.com', 'example.html'), ...]
with ProcessPoolExecutor(max_workers=8) as p:
    p.map(urlretrieve, *zip(*url_dests))
```

The main difference is `map`, like the inbuilt Python function, wants each argument as a separate list, where `starmap` wants a list of argument lists.
We use the standard via `*zip(*)` trick to convert between the two forms.

Then moving from processes to threads is as simple as replacing the `ProcessPoolExecutor` with a `ThreadPoolExecutor`.
Interestingly when using this with the Pyathena library I found that the `ThreadPoolExecutor` didn't seem to result in a concurrency speedup (it was as slow as executing the queries serially).
I'm not yet sure why this is (something in the GIL?); but it makes me want to be more careful when switching to a ThreadPool.

This is a good starting point for more complex workflows.
However if you start using a execute/wait paradigm make sure that you [explicitly raise exceptions](/python-futures-exception).