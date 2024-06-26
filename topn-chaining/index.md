---
categories:
- python
- pandas
- r
date: '2020-03-22T23:34:56+11:00'
image: /images/pandas.png
title: Second most common value with Pandas
---

I really like [method chaining](https://towardsdatascience.com/the-unreasonable-effectiveness-of-method-chaining-in-pandas-15c2109e3c69) in Pandas.
It reduces the risk of typos or errors from running assignment out of order.
However some things are really difficult to do with method chaining in Pandas; in particular getting the second most common value of each group.

This is much easier to do in R's dplyr with its consistent and flexible syntax than it is with Pandas.

# Problem
For the table below find the total frequency and the second most common value of y by frequency for each x (in the case of ties any second most common value will suffice).

|  x |  y | frequency |
| --- | --- | --- |
|  1|  1  | 3 |
|  1|  2  | 2 |
|  1|  3  | 2 |
|  2|  2  | 2 |
|  2|  1  | 1 |
| 3 | 1  | 1 |


Answer:

| x | frequency | second |
| --- | --- | --- |
|  1  | 6   | 2 |
|  2  | 3  |  1 |
|  3  | 1  | NA |

## Solution with dplyr

This is pretty straightforward to solve in R with dplyr; we can first sort the columns by frequency and pick the second element:

```R
d <- data.frame(x=c(1,1,1,2,2,3),
                y=c(1,2,3,1,2,1),
                n=c(3,2,1,1,2,1))

d %>%
group_by(x) %>%
arrange(desc(n)) %>%
summarise(n = sum(n), second = nth(y, 2))
```

This gives exactly the result above.

## Solving with Pandas

Starting with the same dataframe

```python
df = pd.DataFrame({'x': [1, 1, 1, 2, 2, 3],
                   'y': [1, 2, 3, 1, 2, 1],
                   'n': [3, 2, 1, 1, 2, 1]})
```

To get the results over multiple lines is straightforward:

```python
totals = df.groupby('y').n.sum()
# Note nth is 0 indexed
second = df.sort_values('n', ascending=False).groupby('x').y.nth(1)
ans = pd.DataFrame({'n': totals, 'second': second})
```

This is fine, but it means you have to break a chain.
You can chain directly with agg if we want to find the *top* value:

```python
(df
.sort_values('n', ascending=False)
.groupby('x')
.agg(n=('n', 'sum'), first=('y', 'first'))
)
```

Unfortunately there's no built in `second` function.
There is an nth function, but there's no way to pass the argument n in the `agg` call.

We could try to wrap nth in a partial, but I can't work out *where* in pandas nth is defined.
Passing `pandas.core.groupby.generic.DataFrameGroupBy.nth` to agg gives an error.

```python
> df.groupby('y').agg(a=('x', lambda x: nth(x,1)))
TypeError: n needs to be an int or a list/set/tuple of ints
```

We could try to define our own function to find the nth item, `iloc` *almost* works, but if there's an item that doesn't have an nth item it raises an IndexError.

```python
> (df
.sort_values('n', ascending=False)
.groupby('x')
.agg(n=('n', 'sum'), second=('y', lambda y: y.iloc[1]))
)
IndexError: single positional indexer is out-of-bounds
```

Another strategy would be to slice into a running count; in dplyr:

```R
d %>%
group_by(x) %>%
arrange(desc(n)) %>%
mutate(rn = row_number(),
       second = ifelse(rn == 2, y, NA)) %>%
summarise(n=sum(n), second=first(na.omit(second)))
```

We can do this in Pandas because the `first` function ignores NaN values where it can.
Without chaining in Pandas this looks like:

```python
df['rn'] = df.sort_values('n', ascending=False).groupby('x').cumcount()
df['second'] = df.y[df.rn == 1]
df.groupby('x').agg(n=('n', 'sum'), second=('second', 'first'))
```

I'm still not sure how to chain either of the two ways!
One way I can get it to chain is by setting the index and assigning:

```python
(df
.sort_values('n', ascending=False)
.set_index('x')
.assign(second=lambda df: df.groupby('x').y.nth(1))
.groupby('x')
.agg(n=('n', 'sum'), second=('second', 'first'))
)
```

Unfortunately here the cure is worse than the disease and the chain is hard to manage and unreadable.

A cleaner way is to define an `nth` function that does what we need:

```python
import numpy as np
def get_nth(n):
  def nth(x):
    return x[n] if len(x) > n else np.nan
  return nth

(df
.sort_values('n', ascending=False)
.groupby('x')
.agg(n=('n', 'sum'), second=('y', get_nth(1)))
)
```

Another even better option suggested by [Samuel Oranyeli](https://samukweku.github.io/data-wrangling-blog/) is to use `pipe` to be able to use `nth` with other aggregations:


```python
df
.groupby('x')
.pipe(lambda df: pd.DataFrame({'frequency' : df.n.sum(),
                               'second' : df.y.nth(1)}))
```


While these will do, they're still quite frustrating to use.
I'll be watching the Python libraries that are built on top of Pandas, like [Siuba](https://github.com/machow/siuba/) (an adaptation of dplyr) and [Datatable](https://datatable.readthedocs.io/en/latest/index.html) (an adaptation of the R DataTables library), which may make these transformations easier to do.
