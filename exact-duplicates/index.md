---
categories:
- python
- jobs
- nlp
date: '2020-04-13T07:56:24+10:00'
image: /images/duplicate_ads.png
title: Finding Exact Duplicate Text
---

Finding exact duplicates texts is quite straightforward and fast in Python.
This can be useful for removing duplicate entries in a dataset.
I tried this on the [Adzuna Job Salary Predictions Kaggle Competition](https://www.kaggle.com/c/job-salary-prediction) job ad texts and found it worked well.

Naively finding exact duplicates by comparing every pair would be `O(N^2)`, but if we sort the input, which is `O(N log(N))`, then duplicate items are adjacent.
This scales really well to big datasets, and then the duplicate entries can be handled efficiently with [itertools `groupby`](https://docs.python.org/3.7/library/itertools.html#itertools.groupby) to do something like `uniq`.
In this case we can get the indices of the exact duplicates from a list (adding them with `enumerate`).


```python
from itertools import groupby

def second(x):
    return x[1]
    
def exact_duplicate_indices(items)
    exact_duplicates = []
    for _key, group in groupby(sorted(enumerate(items), key=second), key=second):
        group = list(group)
        if len(group) > 1:
            exact_duplicates.append([item[0] for item in group])
    return exact_duplicates
```

If memory was an issue because the items are really large we could even hash them to make the `sorted` possible.
With the 400,000 ad descriptions, on average 1600 characters long, this took 1.3s on my laptop.

I could then look at the size of the duplicate clusters and investigate them:

```
from collections import counter
duplicates = exact_duplicate_indices(ads)
Counter(len(cluster) for cluster in duplicates)
```

This gave a table of frequencies; most clusters only have one duplicate, but there are a few with many duplicates.

| Cluster Size | Frequency |
| ------------ | --------- |
|    2    |    5836    |
|    3    |    293    |
|    4    |    71    |
|    5    |    26    |
|    6    |    7    |
|    7    |    8    |
|    8    |    5    |
|    9    |    7    |
|    10   |    1    |
|    12   |    1    |
|    13   |    1    |
|    14   |    2    |
|    15   |    1    |
|    24   |    1    |

I then inspected the largest clusters.
In my case I kept them in the same order as a dataframe they came from and could index into them with `iloc`.

```python
# Get the 20 largest clusters
megaclusters = sorted(exact_duplicates, key=len, reverse=True)[:20]

# Look it up in the original dataframe
df.iloc[megaclusters[0]]
```

This let me discover that in most of these cases it's due to being posted in multiple locations.
I verified this with a quick analysis that the size of the cluster was almost the number of unique locations.

```
[(len(megacluster), len(df.iloc[megacluster].LocationRaw.unique())) for megacluster in megaclusters]
```

This is useful for exact duplicat, but what about slight variations?
In this case the largest two clusters were in fact the same ad posted to different sites with a slight variation in the footer text.
I'll look at this in the next part of the series.