---
categories:
- jobs
- nlp
- python
date: '2020-05-11T08:44:48+10:00'
image: /images/different_jobs_similar_path.png
title: Minhash Sets
---

We've found pairs of near duplicates texts in the [Adzuna Job Salary Predictions Kaggle Competition](https://www.kaggle.com/c/job-salary-prediction) using [Minhash](/minhash-lsh).
But many pairs will be part of the same *group*, in an extreme case there could be a group of 5 job ads with identical texts which produces 10 pairs.
Both for interpretability and usability it makes sense to extract these groups from the pairs.

# Extracting the Groups Directly with Union Find

Each band of the LSH consists of buckets of items that may be similar; you could view the buckets as a partition of the corpus of all documents.
We then need to find the finest partition such that if two elements are in any subpartition then they are in the biggest partition.
The algorithm for joining these disjoint sets is called [Union Find](https://en.wikipedia.org/wiki/Disjoint-set_data_structure).

The idea is simple; each element contains a pointer to a "parent" element.
There is one element that contains a pointer to itself, which is called the root element.
A partition is all elements that point to the same root element.

Then finding a partition means walking up the parents to the root.
Unioning two partitions just means setting the root of one to the other.

I implement this in Python keeping track of the parents in a separate dictionary (which starts off empty):

```python
def find(x, parents):
    while parents.get(x, x) != x:
        x = parents[x]
    return x

def union(a, b, parents):
    root_a = find(a, parents)
    root_b = find(b, parents)
    if root_b != root_a:
        parents[root_a] = root_b
        if root_b not in parents:
            parents[root_b] = root_b
```

For large sets these root structures can get quite big and it makes sense to [*compress*](https://en.wikipedia.org/wiki/Disjoint-set_data_structure#Path_compression) or somehow shorten the paths by repointing them to ancestors (which doesn't change the partition structure).
I won't worry about that here.

Here is a convenience method to get all the partitions from the parents dictionary:

```python
from collections import defaultdict
def find_sets(parents):
    sets = defaultdict(list)
    for child in parents:
        root = find(child, parents)
        sets[root].append(child)
    return list(sets.values())
```

So for example:

```python
parents = {}
union(2, 1, parents)
union(5, 3, parents)
union(3, 1, parents)
union(7, 9, parents)
find_sets(parents)
```

Returns two partitions:

```
[[2, 1, 5, 3], [7, 9]]
```

## Combining LSH with Union Find


The [datasketch library](http://ekzhu.com/datasketch/index.html) stores the LSH bands as `hashtables`.
By default it's stored in `_dict` which is a mapping from the joint hash across the rows to the elements with that hash (labelled as they are with `lsh.insert`.
We can then union these with our Union-Find algorithm to produce all sets:

```python
def lsh_similar_sets(minhashes, bands, rows):
    lsh = MinHashLSH(num_perm=num_perm, params=(bands, rows))
    for i, mh in enumerate(minhashes):
        lsh.insert(i, mh)

    parents = {}
    for hashtable in lsh.hashtables:
        for items in hashtable._dict.values():
            items = list(items)
            for i in range(len(items)):
                for j in range(len(items)):
                    if i > j:
                        union(items[i], items[j], parents)
    return find_sets(parents)
```

This works fine, but I've already generated the pairs and calculated their similarity.
I could run Union-Find on these pairs, but another approach is taking a graph view.

# Finding the Similar Ad Groups with Graphs

There's a way to view this from a graph point of view.
We consider each ad as a node, and each pair as an edge on the graph.
Then we want to find the connected components of the graph.
This will yield the same groups as UnionFind.

This is straightforward with [networkx](https://networkx.github.io):

```python
import networkx as nx
G=nx.Graph(similar_pairs)
similar_connected = list(nx.connected_components(G))
```

Underneath this works by a [Breadth First Search](https://networkx.github.io/documentation/networkx-1.9.1/_modules/networkx/algorithms/shortest_paths/unweighted.html#single_source_shortest_path_length), which is effectively similar to UnionFind (just going in multiple directions instead of up a single tree).

Doing this with pairs from the LSH having a 3-Jaccard similarity greater than 0.4 yielded one group with 1749 job ads.
I thought this was curious and found there were some job ads in the group with *no* 3-Jaccard overlap.

To understand how this happened I found the shortest path between the pair and looked at the ads.
This yielded a sequence of 8 ads, where each consecutive pair had a 3-Jaccard similarity between 0.4 and 0.6.
But the first and last ad had no overlap.

```python
path = nx.shortest_path(G, 147786, 373053)
edges = list(zip(path, path[1:]))
```

The first 6 job ad pairs were all for an Automotive, Motor Trade Job from "Perfect Placement UK Ltd", a recruitment firm, with some minor differences; likely using the same template for different roles (and they have the same footer).
The seventh pair looks like an edit of the job ad was posted by someone other than that recruitment firm, with mostly the same details.
The first and eighth job ad are completely different, which is quite problematic!

I can think of two use cases of similar job ads.
For getting identical job ads I would need to increase the threshold so these kinds of overlaps don't occur.
For trying to get ads from the same company I would probably look for other heuristics (like contains a long common substring) or prune the edges of the graph rather than taking the whole connected component.
