---
categories:
- python
date: '2020-08-16T20:53:44+10:00'
image: /images/python_lambda.svg
title: Python is not a Functional Programming Language
---

Python is a very versatile multiparadigm language with a great ecosystem of libraries.
However it is *not* a functional programming lanugage, as I know some people have described it.
While you can write it in a functional style it goes against common practice, and has some practical issues.

There is no fundamental definition of a functional programming language but two core concepts are that data are immutable and the existence of higher order functions.
Functional languages like Haskell, Clojure and F# differ in many ways, but tend to default to immutable data and use patterns to create new functions from existing functions.

In Python most data are mutable by default.
While there are some immutable data types like tuple and frozenset, the very commonly used arrays and dictionaries are mutable.
These are often the sources of bugs, such as setting an array as the default argument to a function which is unexpectedly modified.
It also means that if you pass an array or a dictionary to a function there is always a risk that the function modifies it, which makes it harder to reason about the program.

Similarly Python objects are *very* mutable and open.
There's no such thing as private or final variables; just conventions around naming and some guardrails with [`__setattr__`](https://docs.python.org/3/reference/datamodel.html#object.__setattr__), but these can always be overridden.
This makes Python extremely hackable; you can easily "monkeypatch" objects in a couple of lines of code (as opposed to Java where you'd have to create new files with loads of boilerplate).
However this also makes it very hard to understand execution, and there's always a risk that passed data will be modified.

It is always possible to write pure functions in Python, which is [useful for testing](/property-based-testing) and makes the programs easier to understand.
But this requires a lot of discipline and is not the default way to write functions.

Python does have first class fictions and support for higher order functions; however it's unusual to use them extensively and you will hit some barriers if you do.
Python allows easy creation of functions with `def` or with `lambda`, and they can always be passed as arguments.
Python provides `map` in the standard language, and [functools](https://docs.python.org/3/library/functools.html) provides more like `partial` (currying), `reduce` and there are more complete libraries like [toolz](https://toolz.readthedocs.io/en/latest/api.html).

However composing higher order functions is unusual in Python, and you're much more likely to see a list comprehension than a `map`.
Probably because of this the tooling is weaker too.
Functions defined at inner levels, or returned from functions can't be pickled, and this can lead to issues with [multiprocessing](/multiprocess-download).
The resulting functions lose all associated information, such as docstrings and the name of the functions they came from, which makes them very hard to understand compared with explicit functions.
Finally when something goes wrong you end up having to [debug](/pdb) these undocumented functions and the stack trace becomes very hard to interpret; it's really hard to determine where the error occurred.

While you can use Python as a functional language, the same way you can use Java as a functional language, it's not really one.
Whereas I'd say Numpy, Pytorch and Tensorflow are real array programming DSLs in Python, I haven't seen anything I'd want to work with in Python.
This isn't a problem - just use each language to it's strengths and the right tool for the job.