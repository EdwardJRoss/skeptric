---
categories:
- pandas
- python
date: '2021-05-07T07:12:39+10:00'
image: /images/pandas_to_dictionary.png
title: Fast Pandas  DataFrame to Dictionary
---

Tabular data in Pandas is very flexible, but sometimes you just want a key value store for fast lookups.
Because Python is slow, but Pandas and Numpy often have fast C implementations under the hood, the way you do something can have a large impact on its speed.
The fastest way I've found to convert a dataframe to a dictionary from the columns keys to the column value is:

```python
df.set_index(keys)[value].to_dict()
```

The rest of this article will discuss how I used this to speed up a function by a factor of 20.

# Diagnosing and improving a slow implementation

I had a function that performed a few transformations to extract information from a dataframe, but is was pretty slow taking around a second per thousand rows.
I was experimenting in a Jupyter notebook, and came across [a good article on profiling in Jupyter notebooks](http://gouthamanbalaraman.com/blog/profiling-python-jupyter-notebooks.html).
To profile the function `func` with arguments `args` I could run `%prun func(args)`; and the first few rows looked like this:

```
   ncalls  tottime  percall  cumtime  percall filename:lineno(function)
  6307956    1.465    0.000    2.390    0.000 {built-in method builtins.isinstance}
   126008    1.447    0.000    1.802    0.000 {pandas._libs.lib.infer_dtype}
   125987    1.075    0.000   19.294    0.000 series.py:238(__init__)
```

I didn't find this terribly illuminating, but it seemed to be spending a disproportionate time in Pandas and guessing datatypes, which really should not have been a difficult problem.
Because I had a few lines of Python functions it wasn't immediately obvious where this was occurring, so I ran the line profiler, installing it from a Jupyter notebook with `pip install line_profiler` and loading the Jupyter extension with `%load_ext line_profiler`.
Then I could look at the lines taking the most time with the function using `%lprun -f func func(args)`, to find 99% of the time was spent in the following line:

```python
mapping = {tuple(k): v for (_idx, k), v in zip(keys_df.iterrows(), values)}
```

This was my clumsy way to go from a DataFrame to a dictionary; `keys_df = df[keys]` and `values = df[value]`.
The only place I can see that a series would come up is from `iterrows` which emits an index and a series, and the series needs to hold all of the keys.
To do this it needs to work out the least common denominator type of the types of each of the keys (for example if some are integers and some are strings then the resulting series will have dtype object).
And it seemed to do this calculation for every row which was taking a ton of time!

When something like this happens in Pandas or Numpy the best first step is to look for an inbuilt way of doing this, which is likely to be an order of magnitude faster.
A little searching showed that a Pandas `Series` has a `to_dict` method, mapping the index to the values.
So I could replace the line above by the simple expression at the top of the article to make the function go from taking tens of seconds to under a second:

```python
df.set_index(keys)[value].to_dict()
```

I admit I haven't used profiling much before in Python (mostly just manually profiling by typing functions), but it's very easy and useful, especially with line_profiler.