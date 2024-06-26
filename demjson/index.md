---
categories:
- python
- data
date: '2020-07-05T08:00:00+10:00'
image: /images/demjson.png
title: Demjson for parsing tricky Javascript Objects
---

Modern Javascript web frameworks often embed the data used to render each webpage in the HTML.
This means an easy way of extracting data is capturing the string representation of the object [with a pushdown automoton](/parsing-escaped-strings) and then parsing it.
Python's inbuilt `json.loads` is effective, but won't handle very dynamic Javascript, but [demjson](https://github.com/dmeranda/demjson) will (another, much faster alternative is [Chompjs](/chompjs).

The problem shows up when using `json.loads` as the following obscure error:

```
json.decoder.JSONDecodeError: Expecing value: line N column M (char X)
```

Looking at the character in my case looking near the character I see that it is a JavaScript [undefined](https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/undefined), which is not valid in JSON.

```
{"key": undefined, ...
```

However it turns out the `demjson` library handles this well using `demjson.decode(text)`.
It represents undefined with a special `demjson.undefined` class.
Because this isn't serializable I need to convert it to something else; I can walk the dictionary to turn it into a Python `None`.

```
def undefined_to_none(dj):
    if isinstance(dj, dict):
        return {k: undefined_to_none(v) for k, v in dj.items()}
    if isinstance(dj, list):
        return [undefined_to_none(k) for k in dj]
    elif dj == demjson.undefined:
        return None
    else:
        return dj
```

Using demjson and converting `undefined` to `None` works well, but it seems to run about 20 times slower than `json.loads`.
So I'll try a strategy of first using `json.loads` and falling back to `demjson` when necessary.


```python
try:
    data = json.loads(text)
except json.decoder.JSONDecodeError:
    logging.warning('Defaulting to demjson')
    data = demjson.decode(text)
    data = undefined_to_none(data)
```

An alternative approach would be to extend the automoton that extracts the object to replace undefined, and then just parse with `json.loads`.
I'm not sure whether there are other types of non-JSON objects demjson can parse too.