---
categories:
- python
- data
date: '2021-04-28T21:56:42+10:00'
image: /images/chompjs.png
title: Chompjs for parsing tricky Javascript Objects
---

Modern Javascript web frameworks often embed the data used to render each webpage in the HTML.
This means an easy way of extracting data is capturing the string representation of the object [with a pushdown automoton](/parsing-escaped-strings) and then parsing it.
Sometimes Python's `json.loads` won't cut it for dynamic JSON; one option is [demjson](/demjson) but another much faster option is [chompjs](https://github.com/Nykakin/chompjs).

Chompjs converts a javascript string into something that `json.loads`.
It's a little less strict than demjson; for example `{"key": undefined}` will be converted by `chompjs.parse_js_object` to `{"key": "undefined"}` (contrast with `demjson` `{"key": demjson.undefined}` which preserves the type).
However it's much faster, about 20x on the tests I've done, which makes it a much better drop-in replacement for `json.loads` on messy data.