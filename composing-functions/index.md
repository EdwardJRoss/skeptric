---
categories:
- programming
- python
- r
date: '2020-12-06T19:53:54+11:00'
image: /images/magrittr.png
title: Composing Functions
---

R core looks like it's [getting a new pipe operator `|>`](https://developer.r-project.org/blosxom.cgi/R-devel/NEWS/2020/12/04#n2020-12-04) for composing functions.
It's just like the existing [magrittr](https://magrittr.tidyverse.org/) pipe `%>%`, but has been implemented as a syntax transformation so that it is more computationally efficient and produces better stack traces.
The pipe means instead of writing `f(g(h(x)))` you can write `x |> h |> g |> f`, which can be really handy when changing dataframes.

Python's Pandas library doesn't have this kind of convenience and it opens up a class of error that won't happen in that R code.
Here's a typical bit of Pandas code:

```
df_clean = df_raw[(df_raw['colour'] == 'blue') & (df_raw['price'] > 50)]
df_clean.loc[df_clean['price'].isna(), 'price'] = df_clean['price'].mean()
```

There are so much repetition here it's easy to make a mistake.
On the first line `df_raw` is typed *3* times, a typo putting in a different dataframe will lead to subtle runtime errors that are hard to pick up; I've debugged them in my own code many times.
The second line has a similar problem where `df_clean` is typed 3 times (if `df_raw` was put there by accident it could lead to an error).
There's also other Pandas traps here; forgetting the brackets on the first line will lead to an error due to the precedence of &, and I don't know whether the second line actually changes `df_raw` (I may see some warning about that, and then if I want to preserve `df_raw` I'll put a `.copy()` in.

In R dplyr it's much cleaner and you can't accidentally type the wrong thing because we're chaining (here using the magrittr `%>%`, but soon we will be able to use `|>`):

```
df_clean <- df %>%
  filter(colour == "blue", price > 50) %>%
  assign(price = ifelse(is.na(price), 50, mean(price)))
```

In Pandas you can use method chaining (and tools like the [pandas pipe](/pandas-pipe)) to clean it up.
Using [`query`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.query.html) we can get something close to dplyr, but it's still a bit clunky and query can be very slow:

```
df_clean = (df_raw
  .query('colour == "blue" & price > 50')
  .assign(price = lambda df: df['price'].fillna(df['price'].mean()))
```

However there are cases where it's really hard to do in Pandas, like [getting the second most common value in a group](/topn-chaining).
Because Pandas is built by appending functions to the Dataframe class if there's not a method for it you have to patch it in like [pyjanitor does](https://pyjanitor.readthedocs.io/), but it you do a [proper pandas extension](https://pandas.pydata.org/pandas-docs/stable/development/extending.html) it's quite verbose.
In R because it uses a functional approach you can easily reuse common functions rather than having to write (and remember the names of!) Dataframe specific ones.

I think method chaining is a useful way to write data transformations; it exists in most functional languages Haskell, OCaml, F# and in [Clojure's useful threading macros](https://clojure.org/guides/threading_macros).
It's even in Julia and there's a proposal for Javascript.
You can implement it in Python in some sense using magic methods for infix operators, like [Thinc's combinators](https://thinc.ai/docs/api-layers#combinators) but it's against the grain in Python [which s not a functional language](/python-not-functional).