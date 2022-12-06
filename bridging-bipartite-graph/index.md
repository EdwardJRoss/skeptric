---
categories:
- data
- presto
- athena
date: '2020-05-26T15:50:02+10:00'
image: /images/bipartite_matrix.png
title: Bridging Bipartite Graph
---

When you have behavioural data between actors and events you naturally get a bipartite graph.
For example you can have the actors as customers and events as products that are purchased, or the actors as users of a website and the events as videos that are viewed, or the actors as members of a forum and the events as posts they comment on.
One of [the ways](/recommendation-graph) to represent this is to relate actors by the number of events they both participate in.
For example two customers are related by the number of products they have both purchased, or two users by the number of videos they have both viewed, or two forum members by the number of posts they have both commented on.

Mathematically if the adjacency matrix of the bipartite graph is *A*, then this joint matrix is $A A^{T}$.
However most of the time this matrix is very sparse, and calculating this matrix product directly is very slow.
But by iterating over the sorted events you can calculate it efficiently.

Suppose you have the adjacency matrix stored as a list `edges` of pairs of `(actor, event)`.
The naive way to combine them in Python would be:

```python
from collections import defaultdict
counts = defaultdict(int)
for actor_1, event_1 in edges:
  for actor_2, event_2 in edges:
    if event_1 == event_2:
      counts[(actor_1, actor_2)] += 1
```

However this is quadratic in the number of edges.
Instead we could use something like this, which will be much more efficient.

```python
from itertools import groupby
counts = defaultdict(int)
# Sort by the events
sorted_edges = sorted(edges, key=lambda x: x[1])
for _event, group in groupby(sorted_edges, lambda x: x[1]):
  actors = [x[0] for x in group]
  for actor_1 in actors:
    for actor_2 in actors:
      counts[(actor_1, actor_2)] += 1
```

An extension of this would be to filter out any bad actors that participate in too many events (because they contribute very heavily to the counts).

The same thing can be done in Presto SQL; suppose that we have a table `edges` that contains columns `actor, event`.
Then the naive solution is:

```sql
select a.actor as actor_a, b.actor as actor_b
from edges a
join edges b on a.event = b.event
```

Whereas we can do the more efficient version in Presto/Athena SQL using arrays, and even filter on bad actors:

```sql
select actor_a, actor_b, count(*) as count from (
select event, array_agg(actor) as actors
from edges
group by event
-- Filter out any actors with more than 100 events
having cardinality(array_agg(actor)) < 100
)
cross join unnest(actors) as ta(actor_a)
cross join unnest(actors) as ta(actor_b)
group by 1, 2
```

So now you have a relatively efficient way to bridge sparse bipartite graphs, that you can then use for example in [community detection](/community-detection).
