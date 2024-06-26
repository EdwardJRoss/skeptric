---
categories:
- data
date: '2020-06-30T08:00:00+10:00'
image: /images/multiprocessing.png
title: Accelerating downloads with Multiprocessing
---

Downloading files can often be a bottleneck in a data pipeline because network I/O is slow.
A really simple way to handle this is to run multiple downloads in parallel accross threads.
While it's possible to deal with the unused CPU cycles using asynchronous processing, in Python it's generally easier to throw more threads at it.

Using multiprocessing can be very simple if you can turn make the processing occur in a pure function or object method, and both the variables are results are picklable.
Multiprocessing spins up separate threads and passes objects between the threads by pickling them.
This means it's a poor fit if you're executing on or returning large objects, but for sending URLs to fetch and responses this is normally adequate.
Similarly these are simple data structures which are simple to pickle.

Suppose we start with code like this to download files:

```python
from urllib.request import urlretrieve

url_dests = [('http://example.com', 'example.html'), ...]
for url, dest in url_dests:
    urlretrieve(url, dest)
```

Then it's straightforward to add threads with map.
For example with 8 threads:

```python
from multiprocessing import Pool
from urllib.request import urlretrieve

url_dests = [('http://example.com', 'example.html'), ...]
with Pool(8) as p:
    p.starmap(urlretrieve, url_dests)
```

That's all there is to it.
If you need more complicated behaviour it's worth [reading the docs](https://docs.python.org/3/library/multiprocessing.html#multiprocessing.pool.Pool.map) to see alternatives like `imap` and `starmap_async`.

One other thing to keep in mind is the function needs to be picklable, so it can't be a lambda.
If you need to pass custom parameters the best way is with an object.

For example if the original code is:

```python
import os.path
def download(source, dir):
    dest = get_filename(dir, source)
    urlretrieve(source, dest)
    
for url in urls:
    download(url, DIR)
```

You could rewrite it into an object:

```python
from multiprocessing import Pool

class Downloader():
  def __init__(self, dir):
    self.dir = dir
    
  def download(self, source):
    dest = get_filename(self.dir, source)
    urlretrieve(source, dest)
    
downloader = Downloader(DIR)
with Pool(8) as p:
    p.map(downloader.download, urls)
```


Note that this is only safe if the objects are immutable; you can't rely on communication accross threads.
All of this feels a bit flaky, but it's quite practical and effective if you can deal with the limitations.