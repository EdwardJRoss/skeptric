---
categories:
- python
- pandas
date: '2021-01-29T08:00:00+11:00'
image: /images/flatten_object.png
title: Flattening Nested Objects in Python
---

Sometimes I have nested object of dictionaries and lists, frequently from a JSON object, that I need to deal with in Python.
Often I want to load this into a Pandas dataframe, but accessing and mutating dictionary column is a pain, with a whole bunch of expressions like `.apply(lambda x: x[0]['a']['b'])`.
A simple way to handle this is to *flatten* the objects before I put them into the dataframe, and then I can access them directly.
We can automatically assign keys by joining the accessors by a separator, such as "_", so then `x[0]['a']['b']` becomes `x["0_a_b"]`.

Here is a simple recursive function `flatten_object` to do this:


```python
from collections.abc import Iterable
import types
from typing import Any, Dict

def flatten_object(nested: Any, sep: str="_", prefix="") -> Dict[str, Any]:
    """Flattens nested dictionaries and iterables

    The key to a leaf (something is not list-like or a dictionary)
    is the accessors to that leaf from the root separated by sep
    prefixed with prefix.

    If flattening results in a duplicate key raises a ValueError.

    For example:
      flatten_object([{'a': {'b': 'c'}}, [1]],
                     prefix='nest_') == {'nest_0_a_b': 'c', 'nest_1_0': 1}
    """
    ans = {}

    def flatten(x, name=()):
        if isinstance(x, dict):
            for k,v in x.items():
                flatten(v, name + (str(k),))
        elif isinstance(x, Iterable) and not isinstance(x, (str, bytes)):
            for i, v in enumerate(x):
                flatten(v, name + (str(i),))
        else:
            key = sep.join(name)
            if key in ans:
                raise ValueError(f"Duplicate key {key}")
            ans[prefix + sep.join(name)] = x

    flatten(nested)
    return ans
```

It is possible for keys be ambiguous, as in the case of dictionaries with mixed type keys or containing the separator as keys.
Explicitly consider `{'1': 'a', 1: 'b'}` and `{'a_b': 0, 'a': {'b': 1}}`.
There's no universal way to handle these cases, and so the function raises a `ValueError` when it occurs.
Also note that the function drops empty lists or dictionaries: `flatten_object({'a': []}) == {}`, so quite different objects could have the same flattened form.

However I've found this a convenient way to quickly analyse nested data in Pandas, by flattening each of a list of such nested objects and passing the result to `pandas.DataFrame`.
Then when I refine the code I can build a more specific extractor, or use an [extraction DSL](/json-extraction-dsl).
