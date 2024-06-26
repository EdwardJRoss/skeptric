---
categories:
- python
- pandas
date: '2021-04-23T18:21:36+10:00'
image: /images/quantile_class.png
title: Aggregating Quantiles with Pandas
---

One of my favourite tools in Pandas is `agg` for aggregation (it's a worse version of `dplyr`s `summarise`).
Unfortunately it can be difficult to work with for custom aggregates, like [nth largest value](/topn-chaining).
If your aggregate is parameterised, like quantile, you potentially have to define a function for every parameter you use.
A neat trick is to use a class to capture the parameters, making it much easier to try out variations.

Suppose you have some data on avocado prices containing the `year` and the `price` in a dataframe `df`.
If you want to calculate the 25th percentile of price you could run `df.price.quantile(0.25)`.
If you wanted to calculate the median of price per year you could run `df.groupby('year').agg(med_price=('price', 'median'))`.
But what if you wanted to calculate the 25th percentile of price per year?

You could define a function `percentile25`, but defining all those functions gets annoying and slow if you calculate lots of percentiles.
You could define a function that takes a percentile and returns a percentile function, but these inner functions create [confusing stack traces and can't be pickled](/python-not-functional).

A better solution is to use a class, that can act just like a function using the `__call__` parameter.
This one works on Pandas Dataframes and Series:

```
class Quantile:
    def __init__(self, q):
        self.q = q
        
    def __call__(self, x):
        return x.quantile(self.q)
        # Or using numpy
        # return np.quantile(x.dropna(), self.q)
```

Then to calculate the quartiles of price per year you could run

```
(
df
.groupby('year')
.agg(price_p25 = ('price', Quantile(0.25)),
     price_p50 = ('price', Quantile(0.50)),
     price_p75 = ('price', Quantile(0.75)))
)
```