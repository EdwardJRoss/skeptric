---
categories:
- python
- data
- nlp
date: '2020-04-08T08:00:00+10:00'
image: /images/job_title_bigrams.png
title: Counting n-grams with Python and with Pandas
---

Sequences of words are useful for characterising text and for understanding text.
If two texts have many similar sequences of 6 or 7 words it's very likely they have a similar origin.
When splitting apart text it can be useful to keep common phrases like "New York" together rather than treating them as the separate words "New" and "York".
To do this we need a way of extracting and counting sequences of words.

To find all sequences of [n-grams](https://en.wikipedia.org/wiki/N-gram); that is contiguous subsequences of length n, from a sequence `xs` we can use the following function:

```python
def seq_ngrams(xs, n):
    return [xs[i:i+n] for i in range(len(xs)-n+1)]
```

For example:

```python
> seq_ngrams([1,2,3,4,5], 3)
[[1,2,3], [2,3,4], [3,4,5]]
```

This works by iterating over all possible starting indices in the list with `range`, and then extracting the sequence of length `n` using `xs[i:i+n]`.

In the specific case of splitting text into sequences of words this is called [`w-shingling`](https://en.wikipedia.org/wiki/W-shingling) and can be done by splitting:

```python
def shingle(text, w):
    tokens = text.split(' ')
    return [' '.join(xs) for xs in seq_ngrams(tokens, w)]
```

Then to count the `w-shingles` in a corpus you can simply use the inbuilt Counter:

```python
from collections import Counter
def count_shingles(corpus, w):
    return Counter(ngram for text in corpus for ngram in shingle(text, w))
```

If you're dealing with very large collections you can drop in replace Counter with the approximate version [bounter](https://github.com/RaRe-Technologies/bounter).

The rest of this article explores a slower way to do this with Pandas; I don't advocate using it but it's an interesting alternative.

# Counting n-grams with Pandas

Suppose we have some text in a Pandas dataframe `df` column `text` and want to find the `w-shingles`.

```
                                                         text
0                                 Engineering Systems Analyst
1                                     Stress Engineer Glasgow
2                            Modelling and simulation analyst
```

This can be turned into an array using `split` and then unnested with [`explode`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.explode.html#pandas.Series.explode).

```python
words = (df
.text
.str.split(' ')
.explode()
)
```

This would result in one word per line.
The index is preserved so you can realign it with the original series.

```
0         Engineering 
0         Systems
0         Analyst
1         Stress
1         Engineer
1         Glasgow
2         Modelling
2         and
2         simulation
2         analyst
```

To get sequences of words you can use the [`shift`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.shift.html) operator which is like `lead` and `lag` in SQL.

```python
next_word = words.groupby(level=0).shift(-1)
```

Resulting in:

```
0         Systems
0         Analyst
0         NaN 
1         Engineer
1         Glasgow
1         NaN
2         and
2         simulation
2         analyst
2         NaN
```

and these can be recombined with `(words + next_word).dropna()`:

```
0         Engineering Systems
0         Systems Analyst
1         Stress Engineer
1         Engineer Glasgow
2         Modelling and
2         and simulation
2         simulation analyst
```

Finally you can find the total with `value_counts`.

While this is a bit messier and slower than the pure Python method, it may be useful if you needed to realign it with the original dataframe.
This can be abstracted to arbitrary n-grams:

```python
import pandas as pd
def count_ngrams(series: pd.Series, n: int) -> pd.Series:
    ngrams = series.copy().str.split(' ').explode()
    for i in range(1, n):
        ngrams += ' ' + ngrams.groupby(level=0).shift(-i)
        ngrams = ngrams.dropna()
    return ngrams.value_counts()    
```

This is similar to the approach of the R `tidytext` library for [extracting n-grams](https://www.tidytextmining.com/ngrams.html) which has the function `unnest_tokens` that can produce `ngrams` of arbitrary length.