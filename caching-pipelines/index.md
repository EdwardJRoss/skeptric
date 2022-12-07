---
categories:
- data
date: '2020-06-29T08:00:00+10:00'
image: /images/donefile.png
title: Caching Data Pipelines
---

Data pipelines can often be thought of as a chain of pure functions passing data between them, even if they are not implemented that way.
As long as you can access the sources you can always rerun the pipeline from end to end.
However if the processing takes a long time to run it can be convenient to cache some step as a sink, especially if steps are likely to fail.

A common problem I have is after a process has been running for a long time it fails on some malformed data.
Depending on the problem the data can normally be repaired, filtered or the process extended to resolve it.
However it can take a long time to bisect the problem, and all the intermediate calculation is wasted.
If the good data that has been processed can be saved then it can save hours of recalculation; and can be deleted and recalculated if the processing changes.

A classic way to do this is with file based mechanisms.
A pipeline can be decomposed into steps that each consume files and emit files.
Then if one stage of the pipeline fails you only need to repair and rerun that stage.
There are many ways to orchestrate this from Makefiles to Airflow.

This can be make more granular if you can break the processing up into pieces that are each serialised to separate files.
In some applications where you are integrating diverse datasources there is a natural separation.
This also means you can continue to run the rest of the pipeline on the data you can process.
If the output file is already there you just skip the step (and if you change the pipeline you need to delete the output files).

```python
output_dir = pathlib.Path(...)
path = output_dir / key
if path.exists():
    continue
...
write(data, path)
```

Note that it's important to write to the file in one statement; having incomplete data written to this file would mean it never gets fixed and leads to more problems.
If you need to serialise data as you go along you could do it at a directory level and have an empty `DONE` file to indicate the process was successfully completed.

An interesting way to do this at a more granular level with libraries like [shelve](https://docs.python.org/3/library/shelve.html) or [diskcache](http://www.grantjenks.com/docs/diskcache/tutorial.html).
These allow you to back a dictionary with a filestore or a SQLite database respectively.
For an expensive computation you could use this to memoise a function, for example:

```python
def transform(key):
  with shelve.open('transform') as cache:
    if key not in cache:
      cache[key] = _transform(key)
    return cache[key]
```

It could be useful to make this a decorator like [memozo](https://github.com/sotetsuk/memozo) and [percache](https://pypi.org/project/percache/).
Though diskcache seems like the strongest solution; being process safe for parallel processing, and implementing expiration (useful for infrequently updated resources) among other things.