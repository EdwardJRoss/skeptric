---
categories:
- sicp
date: '2020-09-30T08:00:00+10:00'
image: /images/sicp.jpg
title: SICP Exercise 1.1
---

Exercise from SICP:

[Exercise 1.1.](https://mitpress.mit.edu/sites/default/files/sicp/full-text/book/book-Z-H-10.html#%_thm_1.1)  Below is a sequence of expressions. What is the result printed by the interpreter in response to each expression? Assume that the sequence is to be evaluated in the order in which it is presented.

```
10
(+ 5 3 4)
(- 9 1)
(/ 6 2)
(+ (* 2 4) (- 4 6))
(define a 3)
(define b (+ a 1))
(+ a b (* a b))
(= a b)
(if (and (> b a) (< b (* a b)))
    b
    a)
(cond ((= a 4) 6)
      ((= b 4) (+ 6 7 a))
      (else 25))
(+ 2 (if (> b a) b a))
(* (cond ((> a b) a)
         ((< a b) b)
         (else -1))
   (+ a 1))
```

# Solution

We can step through these using the substitution model with environment.
You can actually do this in Dr Racket using [the Stepper](https://docs.racket-lang.org/stepper/)

* `10` evaluates to 10
* `(+ 5 3 4)` is 5+3+4 which is 12
* `(- 9 1)` is 9-1 which is 8
* `(/ 6 2)` is 6 / 2 which is 3
* `(+ (* 2 4) (- 4 6))` using substitution model becomes `(+ 8 _2)` which is 6.
* `(define a 3)` has no result, updates the environment to have `a` as 3
* `(define b (+ a 1))` substitutes to `(define b (+ 3 1))` after retrieving `a` from the environment, which substitutes to `(define b 4)` which updates the environment
* `(+ a b (* a b))` substitutes from environment as `(+ 3 4 (* 3 4))` which substitutes to `(+ 3 4 12)` which is 19.
* `(= a b)` substitutes from the environment as `(= 3 4)` which results in false, `#f`.

## Compound if

The expression

```
(if (and (> b a) (< b (* a b)))
    b
    a)
```

substitutes to

```
(if (and (> 4 3) (< 4 (* 3 4)))
    4
    3)
```

which is

```
(if (and (> 4 3) (< 4 (* 3 4)))
    4
    3)
```

which substitutes to


```
(if (and (> 4 3) (< 4 12))
    4
    3)
```

and in turn becomes


```
(if (and #t #t)
    4
    3)
```

which then evaluates to


```
(if #t
    4
    3)
```

finally yielding 4.

## Another example
    
```
(cond ((= a 4) 6)
      ((= b 4) (+ 6 7 a))
      (else 25))
```

Injecting from the environment that `a` is 3 and `b` is 4 gives;

```
(cond ((= 3 4) 6)
      ((= 4 4) (+ 6 7 3))
      (else 25)
)
```

which, in the substitution model is the same as

```
(cond (#f 6)
      (#t 16)
      (else 25)
)
```

Which results in the first true branch, 16.
      
## Applying on an if

We substitute to get the following sequence of partial evaluations; using that `a` is 3 and `b` is 4:

```
(+ 2 (if (> b a) b a))
(+ 2 (if (> 4 3) 4 3))
(+ 2 (if #t 4 3))
(+ 2 4)
6
```

## Applying on a Cond

We substitute to get the following sequence of partial evaluations; using that `a` is 3 and `b` is 4:

```
(* (cond ((> a b) a)
         ((< a b) b)
         (else -1))
   (+ a 1))

(* (cond ((> 3 4) 3)
         ((< 3 4) 4)
         (else -1))
   (+ 3 1))
(* (cond (#f 3)
         (#t 4)
         (else -1))
   4)
(* 4
   4)
16
```