---
categories:
- python
- programming
date: '2020-08-22T22:52:46+10:00'
image: /images/maybe_monad.png
title: Maybe Monad in Python
---

A *monad* in languages like Haskell is used as a particular way to *raise* the domain of a function beyond where it was domain.
You can think of them as a generalised form of function composition; they are a way of taking one type of function and getting another function.
A very useful case is the [*maybe monad*](https://en.wikipedia.org/wiki/Monad_(functional_programming)#An_example:_Maybe) used for dealing with missing data.

Suppose you've got some useful function that parses a date: `parse_date('2020-08-22') == datetime(2020,8,22)`.
However sometimes `None` will be passed as an argument which leads to an error, but you just want it to return None.
You can explicitly write this as a new function:

```python
def parse_optional_date(date):
  if date is None:
    return None
  else:
    return parse_date(date)
```

This is pretty straightforward to do, but it means that whenever you are writing or using a function you have to think about how it handles None.
Do you want it to be an error, or to pass through?

The approach generally taken in Haskell is that it should be an error, but you can get this behaviour using the Maybe monad.
Explicitly there's a functor (a fancy name for a function that acts on functions in a composable way) that takes a function and extends it like our `parse_optional_date` example.
In Python it would look something like:

```python
def maybe_fmap(f):
  return lambda(x): f(x) if x is not None else None
```

This takes one function and returns a new one that passes through nulls; if we were going to add type annotations it would look something like:

```python
from typing import Callable, TypeVar, Optional
a = TypeVar('a')
b = TypeVar('b')
def maybe_fmap(f: Callable[a, b]) -> Callable[Optional[a], Optional[b]]:
  return lambda(x): f(x) if x is not None else None
```

In Haskell you will generally see Monads defined in terms of the `bind` (`>>=`) operator, which is closely related.

```python
def maybe_bind(x: Optional[a], f: Callable[a, Optional[b]]):
  return f(x) if x is not None else None
```

To be clear, I wouldn't advocate actually doing this because [Python is not a functional language](/python-not-functional), and you're going to end up with some gnarly stack traces and hard to understand functions.
But it's a useful concept to have in mind when designing functions; it's systematically trivial to add support for `None`.

This is just one example of a monad to lift functions.
Another one is `map` which is a functor that raises a function to one that can handle lists.
In Haskell the idea is even used to extend pure functions, that are [easy to test](/property-based-testing), into scenarios where there is I/O or state.
However I'm not convinced the conceptual overhead is always worth it.

If you do want to use Maybe pattern in Python here are some libraries that offer it with increasing levels of sophistication:

* [pymaybe](https://github.com/ekampf/pymaybe) extends a `Maybe` object with dunder methods so a `Nothing` is passed through subsequent operations and is exited with a `get` or `or_else` method
* [maybe-else](https://github.com/matthewgdv/maybe) does the same thing, but is exited with an `else_`
* [returns](https://returns.readthedocs.io/en/latest/index.html) has a general monad implementation, good integration with mypy and some documentation
* [pymonad](https://github.com/jasondelaat/pymonad) has a generic implementation of monads, but not much documentation