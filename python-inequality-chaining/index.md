---
categories:
- python
date: '2020-04-03T07:54:42+11:00'
image: /images/python_inequality.png
title: Python Inequality Chaining
---

In Python the comparison a <= b == c < d does the mathematically correct thing.
This is a handy notational trick.

This wasn't obvious to me because a lot of programming languages treat these associatively, so that a <= b < c may resolve to (a <= b) < c.
This is very dangerous if boolean (True or False) are coerced to integers (1 or 0) because it may *look* like it works but give the wrong results.

However [Python's documentation](https://docs.python.org/3/reference/expressions.html#comparisons) explains that a chained comparison like a <= b < c is translated to (a <= b) and (b < c), which is exactly what you expect (with b only evaluated once, in case it has side effects).
This is a neat trick that can make code a bit easier to read (though you have to be careful if you're switching between languages!)