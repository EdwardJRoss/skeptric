---
categories:
- python
- r
date: '2020-05-14T14:28:44+10:00'
image: /images/cartesian_product.svg
title: Cartesian Product in R and Python
---

You've got a couple of groups and you want to get every possible combination of them.
This is called the [Cartesian Product](https://en.wikipedia.org/wiki/Cartesian_product) of the groups.
There are standard ways of doing this in R and Python.

## Python: List Comprehensions

Concretely we've got (in Python notation) the vectors `x = [1, 2, 3]` and `y = [4, 5]` and we want to get all possible pairs: [(1, 4), (2, 4), (3, 4), (1, 5), (2, 5), (3, 5)]`.
The "pythonic" way to do this is with a list comprehension:

```python
[(x_, y_) for x_ in x for y_ in y]
```

Another possibility is to use [`itertools.product`](https://docs.python.org/3/library/itertools.html#itertools.product) which is especially useful for a large number of lists.


## R: Expand.grid

In R we can use `expand.grid` to get a `data.frame` of all pairs:
```R
expand.grid(x=x, y=y)
```

In this expression the `x` and `y` to the left of the `=` sign are the names of the columns in the dataframe.
I find this really useful when creating plots of functions with `ggplot2` to try every possible combination of parameters.
You can also do this manually using `rep`; for example:

```R
data.frame(x=rep(x, length(y)), y=rep(y, each=length(x)))
```

## Python: More Complex List Comprehensions

What if we have a slightly harder problem: there's another vector `z = [6, 7]` and we want to take every aligned pair from `y` and `z` and combine it with every possible `x`.
So the output should be `[(1, 4, 6), (2, 4, 6), (3, 4, 6), (1, 5, 7), (2, 5, 7), (3, 5, 7)]`.
This is straightforward with list comprehensions by combining `y` and `z` with zip:

```python
[(x_, y_, z_) for x_ in x for y_, z_ in zip(y, z)]
```

This is one of the strengths of Python list comprehensions, it's easy to extend with different variables and with functions acting on those variables.


## R: tidyr expand

I don't know how to do this harder task in R with `expand.grid`, and so I would have to fallback to the long way with `rep`.
This would be

```R
data.frame(x=rep(x, length(y)), y=rep(y, each=length(x)), z=rep(z, each=length(x)))
```

This gets quite tedious to write!

However there are neat ways to do this with the [tidyr](https://tidyr.tidyverse.org/index.html) package, and in particular with the [`expand`](https://tidyr.tidyverse.org/reference/expand.html) function.
You can solve it like this:

```R
expand(data.frame(y=y, z=z), x, nesting(y, z)
```

This gets all combinations of `x`, `y`, and `z`, providing that the pairs `y` and `z` are in the `data.frame` from the first argument.

Note that `expand` is *not* referentially transparent, and the variables rely on their names in the data frame (as is typical of tidyverse functions).
For example `expand(data.frame(y=z, z=y), x, nesting(y, z)` will reverse the order of the last two columns.
