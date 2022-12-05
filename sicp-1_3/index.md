---
categories:
- sicp
date: '2020-10-02T08:00:00+10:00'
image: /images/sicp.jpg
title: Sicp Exercise 1.3
---

Exercise from SICP:

[Exercise 1.2.](https://mitpress.mit.edu/sites/default/files/sicp/full-text/book/book-Z-H-10.html#%_thm_1.3) 

> Define a function that takes three numbers as arguments and returns the sum of the two larger numbers.

# Solution

The first thing we need to do is to get the largest two numbers from 3 numbers.
We can do this with a conditional statement.

```
(define (sum-square-largest-two a b c)
        (cond ((and (<= a b) (<= a c)) (sum-of-squares b c))
              ((and (<= b a) (<= b c)) (sum-of-squares a c))
              ((and (<= c a) (<= c b)) (sum-of-squares a b))))
```