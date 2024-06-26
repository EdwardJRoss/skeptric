---
categories:
- python
date: '2021-01-28T19:18:14+11:00'
image: /images/json_extract_dsl.png
title: Extracting Fields from JSON with a Python DSL
---

Indexing into nested objects of dictionaries and lists in Python is painful.
I commonly come up against this when reading JSON objects, and often fields can be omitted.
I haven't found a solution to this and so I've invented a tiny DSL to do this.

It works like this:

```python
d = [{'a': [{'b': 'c'}, {'d': ['e']}]}]

assert extract(d, '0.a.1.d.0') == d[0]['a'][1]['d'][0]
assert extract(d, '1.a.1.d.0') == None
```

You can specify a path into an object, separated by periods, and it will extract it returning `None` if that path doesn't exist.
The main limitations of this approach are:

* The field separator (`.` by default) can't be used in dictionary keys
* Only strings or integers can be used as dictionary keys
* Strings consisting of integers (e.g. `"1"` or `"-21"`) can't be used as dictionary keys

The implementation is pretty simple, it uses [`itemgetter`](https://docs.python.org/3/library/operator.html#operator.itemgetter) to recursively step through the path.
The only complication is to index into arrays we have to convert strings representing integers into integers (hence the limitation above).

```python
from operator import itemgetter
import re
from typing import Union, Callable, Any

def is_integer(s: str) -> bool:
    return re.match('^-?[0-9]+$', s) is not None

def convert_integers(s: str) -> Union[str, int]:
    if is_integer(s):
        return int(s)
    else:
        return s

def extract(obj: Any, path: str, sep: str='.', default=None) -> Any:
    steps = [convert_integers(x) for x in path.split(sep)]
    for step in steps:
        try:
            return itemgetter(step)(obj)
        except (KeyError, IndexError, TypeError):
            return default
```

A more functional (but [less Pythonic](/python-not-functional)) way to do this is by composing the `itemgetter`s.


```python
import functools

def compose(*functions):
    return functools.reduce(lambda f, g: lambda x: f(g(x)), functions)

def extractor(path: str, sep: str='.') -> Callable[[Any], Any]:
    steps = [convert_integers(x) for x in path.split(sep)]
    return compose(*map(itemgetter, reversed(steps)))

def extract2(obj: Any, path: str, sep: str='.', default=None) -> Any:
  try:
    return extractor(path, sep)(obj)
  except (KeyError, IndexError, TypeError):
    return obj
```

Note that type checking isn't very useful here; this approach is very dynamic and statically verifying a caller is using it correctly would be very hard.
I'm not sure if something like [Haskell's Lens library](https://github.com/ekmett/lens#lens-lenses-folds-and-traversals) solves this; but when dealing with arbitrary JSON it's hard to know what the data will be like anyway.

This gives a simple but effective way to extract fields from structured data.
For example if you were getting JSON-LD or Microdata for a [jobposting](/schema-jobposting) you could extract the currency using something like: `extract(jobposting, "salaryCurrency") or extract(jobposting, "baseSalary.currency")` since it can optionally be put into either field.
