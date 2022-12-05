---
categories:
- jobs
- nlp
- data
- graphs
date: '2020-05-30T20:30:57+10:00'
image: /images/similar_companies.png
title: Finding Duplicate Companies with Cliques
---

We've found [pairs of near duplicate texts](/minhash-lsh) in 400,000 job ads from the [Adzuna Job Salary Predictions Kaggle Competition](https://www.kaggle.com/c/job-salary-prediction).
When we tried to extracted [groups of similar ads](/minhash-sets) by finding connected components in the graph of similar ads.
Unfortunately with a low threshold of similarity we ended up with a chain of ads that were each similar, but the first and last ad were totally unrelated.
One way to work around this is to find *cliques*, or a group of job ad were every job ad is similar to all of the others.

Adzuna gets most of its ads by aggregating ads from other job ad websites.
I have noticed that sometimes and advertiser has slightly different names when it is sourced from different websites.
For example a company is called "360 Rockwool" on jobs sourced from totaljobs.com, and "Rockwool" on jobs from other sources.
It would be useful to be able to identify jobs posted by the same company to better understand the job ad data.

When a job ad is written by a company that regularly recruits people they usually use a standard template that contains things like a description of the company.
So we'd expect the Jaccard similarity of these job ads to be high for a number of job ads.
So the idea is to look for groups of 5 job ads that all have at least 20% 3-Jaccard similarity to each other.
The cutoffs could be tuned but it's a fine place to start.

While this does find some similar companies it also does bring in some noise, like jobs sources to multiple recruiters.
A bigger barrier is the algorithm to find all cliques is exponential and so for larger groups it takes longer than is practicable.

# Finding cliques

From our [pairs of near duplicate texts](/minhash-lsh) we have a list of pairs of job ads with 3-Jaccard similarity of at least 20% in a vector `similar`.
We can then separate it into [connected components with networkx]((/minhash-sets))

```python
import networkx as nx
G = nx.Graph(similar)
similar_connected = sorted(nx.connected_components(G), key=len)
```

We can then extract from these the cases where the group contains at least 5 ads and they correspond to different companies in the source dataframe `df`.

```python
multicompany = [list(idxs) for idxs in similar_connected 
                if len(idxs) >= 5
                and (df
                     .iloc[list(idxs)]
                     .Company
                     .dropna()
                     .unique()
                     .size) > 1]
```

Then we can extract the cliques containing at least 5 members using `find_clique`:

```python
n = 0
SG = G.subgraph(multicompany[n])
cliques = [clique for clique in nx.find_cliques(SG) if len(clique) >= 5]
```

We can then examine the companies in the cliques and the job ad texts.

# Examining similar companies

This technique definitely extracts some similar companies.
There are some interesting examples where a job board name has replaced the company name, or it has been completely removed.
Sometimes the terms or contact details that identify the company have been removed making it difficult to directly extract from the text.
In these cases it can be difficult to determine *what* the source company actually is.

I found most of the examples I looked at were from recruitment companies sourced across multiple job boards.
It's common practice for recruiters to use software to post to multiple job boards, and how their name appears probably depends on how they set up the accounts and integrations, which is why this happened.
Recruiter job ads on Adzuna tend to be fairly bland and not have much company detail, so are hard to identify.

Looking at a bigger clique with 10 ads I found an example where it looks like the same job had been contracted out to two different recruiters.
A substantial portion of the text was the same, with some slight changes, but they came from different recruitment companies.
In this case just because the ads were very similar did not mean the companies were the same.

Between all this it seems it's quite difficult to identify the true company that posted the ad on Adzuna.
On a higher quality dataset the best approach would likely be to try to extract the company name from the ad text because most companies have a section about themselves in the ad.

# Cliques are slow to find

This approach worked well on small graphs, but finding all the cliques in the graph can take exponential time (and there can be exponentially many cliques!).
Once I got to graphs of around 400 nodes I saw this issue.
I found on a graph with 369 nodes it took 3 seconds, with 392 nodes it took 12 seconds, with 427 nodes it took 20 seconds and I'm still waiting on a graph with 429 nodes.

I'm sure there are other heuristic techniques that would work for finding dense groups on the graph that are more efficient, like using mincut to iteratively reduce the graph to a highly connected core.