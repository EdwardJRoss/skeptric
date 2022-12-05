---
categories:
- python
date: '2020-12-18T22:10:03+11:00'
image: /images/topological_ordering.svg
title: Pip Can Now Resolve Dependencies
---

Something that has always bothered me about pip in Python is that you would get errors about inconsistent packages.
Things still seemed to work surprisingly often, but it meant that the order you installed packages could lead to very different results (and one ordering may cause your test to fail, even if that doesn't succeed).

Now there is a [new resolver in Pip 20.3](http://pyfound.blogspot.com/2020/11/pip-20-3-new-resolver.html) for pip that checks the dependencies and tries to find versions that meet all constraints.
This is a huge step forward for Python, and makes it easier to adopt.