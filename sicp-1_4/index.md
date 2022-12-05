---
categories:
- sicp
date: '2020-10-03T08:00:00+10:00'
image: /images/sicp.jpg
title: SICP Exercise 1.4
---

Exercise from SICP:

[Exercise 1.4.](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/6515/sicp.zip/full-text/book/book-Z-H-10.html#%25_thm_1.4) 

> Observe that our model of evaluation allows for combinations whose operators are compound expressions. Use this observation to describe the behavior of the following procedure:

```
(define (a-plus-abs-b a b)
  ((if (> b 0) + -) a b))
```

# Solution

There are two possible branches, if b is positive then we get:

```
((if (> b 0) + -) a b)
((if #t + -) a b)
(+ a b)
```

Wheras if b is non-positive we get

```
((if (> b 0) + -) a b)
((if #f + -) a b)
(- a b)
```

So we always get a added to the absolute value of b.
This is a nice use of a first order function.