---
categories:
- python
date: '2020-11-30T21:55:46+11:00'
image: /images/pandas_pipe.png
title: Chaining with Pandas Pipe function
---

I often use method chaining in pandas, although certain problems like [calculating the second most common value](/topn-chaining) are hard.
A really good solution to adding custom functionality in a chain is Pandas [pipe](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.pipe.html) function.

For example to raise a function to the 3rd power with numpy you could use

```python
np.power(df['x'], 3)
```

But another way with pipe is:

```python
df['x'].pipe(np.power, 3)
```

Note that you can pass any positional or keyword arguments and they'll get passed along.
So `df.pipe(f, *args, **kwargs)` is equivalent to `f(df, *args, **kwargs)`.

If you build up a series of transforms on dataframes to dataframes you can then express this with a series of `pipe` statements.
It even [works with a groupby](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.core.groupby.GroupBy.pipe.html).