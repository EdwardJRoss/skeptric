---
categories:
- sicp
date: '2020-10-04T08:00:00+10:00'
image: /images/sicp.jpg
title: SICP Exercise 1.5
---

Exercise from SICP:

[Exercise 1.5.](https://mitp-content-server.mit.edu/books/content/sectbyfn/books_pres_0/6515/sicp.zip/full-text/book/book-Z-H-10.html#%25_thm_1.5) 

Ben Bitdiddle has invented a test to determine whether the interpreter he is faced with is using applicative-order evaluation or normal-order evaluation. He defines the following two procedures.

```
(define (p) (p))

(define (test x y)
  (if (= x 0)
      0
      y))
```

Then he evaluates the expression

```
(test 0 (p))
```

What behavior will Ben observe with an interpreter that uses applicative-order evaluation? What behavior will he observe with an interpreter that uses normal-order evaluation?

# Solution

## Appicative order

With applicative order the first step is to evaluate `(test 0 (p))`, expanding `(p)` by its definition.
But that evaluates to itself so the program gets stuck in an infinite loop and does not terminate.

## Normal order

We get the evaluation chain:

```
(test 0 (p))
(if (= 0 0) 0 (p))
(if #t 0 (p))
0
```

So it results in 0.
Because we expand the definitions in normal order, and the `if` statement avoids it, we never hit the recursive loop in normal order.