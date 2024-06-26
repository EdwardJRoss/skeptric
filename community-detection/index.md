---
categories:
- data
date: '2020-05-20T22:12:25+10:00'
image: /images/football-sbm-fit.svg
title: Community detection in Graphs
---

People using a website or app will have different patterns of behaviours.
It can be useful to [cluster](/clustering-segmentation) the customers or products to help understand the business and make better strategic decisions.
One way to view this data is as an [interaction graph](/recommendation-graph) between people and the product they interact with.
Clustering a graph of interactions is called "community detection"

Santo Fortunato's [review article](https://arxiv.org/abs/0906.0612) and [user guide](https://arxiv.org/abs/1608.00163) provides a really good introduction to community detection.
The intuitive idea is that you want to group nodes such that a node in the group will on average have more connections to other nodes in the group than to nodes in other groups.
However the devil is in the detail and there's no clear definition of whether one community is "better" than the other.

In fact it's even hard to say how similar two partitions into communities are.
In the article they recommend using the variation of information: $V(X, Y) = H(X \vert Y) + H (Y \vert X)$ where *H* is the Shannon entropy of the cluster assignments.
But a deeper problem is there are rarely ground truths for what the communities are.
There exist some examples in the literature; like the Zacchary Karate Club network about members of a Karate Club that split into two separate clubs, or another about bottlenose dolphins observed together that later migrated to different areas. 
But the datasets are typically small, noisy (a single missed observation could radically change the graph) and scarce and it's hard to say how *true* their ground truths are.

Because of the ambiguity of definition and the rarity of well grounded datasets there are many different techniques to solve the problem, and it's really hard to evaluate which ones are best.
There are methods that try to cut the graph into pieces; either directly or via looking at the spectrum of the Laplacian matrix.
There are optimisation methods that try to maximise some function of how good the partitions are; modularity being a popular one but that has some limitations on how small a community it can detect and has many "near maxima".
There are methods based on statistical inference; most popular are stochastic block models that assume there's a constant probability of connection between each community and tries to infer the communities that maximise the log likelihood.
For stochastic block models the number of communities is a hyperparameter, but there are techniques that address this by setting a Bayesian prior on the number of communities.
Another approach is to look at dynamics on the network like random walks, synchronisation of coupled oscillators or spin glasses and define communities based on the domains (attractors, synchronised groups, final spin state).
There's also label propagation where each node is iteratively assigned to be in the same group as the majority of its neighbours, randomly picked when there is a tie.

There are even meta-methods like consensus clustering which generates a new algorithm from existing clustering algorithms.
The original stochastic clustering algorithm is run a number of times to generate the *consensus matrix* which contains the probability of two vertices being in the same cluster under the clustering algorithm.
This consensus matrix is then thresholded and the the process is run iteratively until the clustering algorithm run multiple times gives the same result every time.

So if there are lots of different ways of creating communities, and they are hard to evaluate where do we start?
It makes sense to start with some domain specific measures on how good the communities are.
For example you might already have some measures on your products or customers that you would expect to be reflected in their behaviour; you should check how tightly the communities segment this behaviour.
You may also have some constraints on the number and size of groups to be a useful grouping.
Finally you may have some internal categorisation of products, or even some intuitive knowledge of what should be similar, that you could use to check the communities against.
None of these are perfect, but you can at least determine whether a community assignment is viable.

Then unless you have an expectation a particular algorithm would work best, I would go through the algorithms that are popular, have good implementations and are efficient enough to complete on your dataset, until you find one that is viable.
For example the [Louvian Algorithm](https://en.wikipedia.org/wiki/Louvain_modularity) for maximising modularity is efficient and has implementations in [Neo4J](https://neo4j.com/docs/graph-algorithms/current/algorithms/louvain/), [Python](https://python-louvain.readthedocs.io/en/latest/api.html) and [R](https://igraph.org/r/doc/cluster_louvain.html), and you can walk down the dendrogram until you get to a number of clusters that looks viable.
Or [Infomap](https://www.mapequation.org/) based on random walks has implementations in [Python](https://mapequation.github.io/infomap/python/) and [R](https://igraph.org/r/doc/cluster_infomap.html).
The [graph-tool](https://graph-tool.skewed.de/) package has a really powerful implementation of [stochastic block models](https://graph-tool.skewed.de/static/doc/demos/inference/inference.html) using MCMC that can deal with multiple types of graphs (and metadata can be seen as a type of graph).
It's worth looking through the algorithms available in [NetworkX](https://networkx.github.io/documentation/stable/reference/algorithms/community.html), [igraph](https://igraph.org/c/doc/igraph-Community.html) and [Neo4J](https://neo4j.com/docs/graph-data-science/current/algorithms/community/). 

While it can seem overwhelming these techniques can reduce an intractable problem of grouping tens of thousands of items into a tractable one of evaluating hundreds of groups.
If you can come up with good heuristic criteria for evaluating the groups you can sample from the buffet of techniques for detecting communitites and see what works in your application.