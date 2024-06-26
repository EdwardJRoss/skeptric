---
categories:
- data
date: '2020-05-05T08:00:00+10:00'
image: /images/cluster.svg
title: Clustering for Segmentation
---

Dealing with thousands of different items is difficult.
When you've got a couple of dozen you can view them together, but as you get into the hundreds, thousands and beyond it becomes necessary to group items to make sense of them.
For example if you've got a list of customers you might group them by state, or by annual spend.
But sometimes it would be useful to split them into a few groups using some heuristic criteria; clustering is a powerful technique to do this.

First you'll need some data with items and features about the items; for example this could be from a customer database or from a survey you have conducted.
In the customer example it could look like:

| customer_id | first_date | state | industry      | acquisition_channel | support_calls | annual_spend |
|-------------|------------|-------|---------------|---------------------|---------------|--------------|
| 1           | 2003-05    | SA    | agriculture   | Referral            | 2             | 10,000       |
| 2           | 2020-03    | Vic   | manufacturing | Direct              | 0             | 3,100        |

# Clustering and Descriptive features

You need to separate your *clustering features* from your *descriptive features*.
The clustering features are things you would segment your customers on, for example it might not make sense to use `support_calls` because this will be highly variable and won't apply to new customers, or maybe the `first_date` isn't really going to be meaningful for customer behaviour.
The descriptive features are the non-clustering columns that are useful for characterising a group.

In this example it might make sense to have `state` and `industry` as clustering features, and `support_calls` and `annual_spend` as descriptive features.
The simplest clustering is a *full pivot* of the clustering features; that is every combination.
In this case we'd treat every state and industry as a separate group.
The problem with this is we may end up with *lots* of groups, and many small groups for uncommon combinations (like customers in the `ACT` in `manufacturing`).
Sometimes you can sweep this away by collecting all the small groups into an `Other` group, but you can lose valuable information this way.

# Similarity measure

Clustering works by grouping together objects that are more similar to each other than those in other groups.
To do this we need to define what "similar" is.
The general way of doing this is to combine distances as a weighted average of features (this is described well in [Elements of Statistical Learning (2nd ed.)](https://web.stanford.edu/~hastie/ElemStatLearn/) section 14.3).

First you need a notion of distance for each feature in the similarity measure.
For example you might have a notion of distances between states (maybe 1 if they share a border or 0 otherwise), and industries (based on subject matter expertise; for example you might curate your own hierarchy).
Otherwise you could just say the distance is 1 if they are the same and 0 if they are different (see [Gower's formula](https://stat.ethz.ch/R-manual/R-devel/library/cluster/html/daisy.html) for a fairly general dissimilarity function).
For annual spend the distance could be the absolute difference between them.

Then you combine them to create a distance with a weighted average accross the features.
Because they are on different scales a weight of 1 does *not* give them all equal importance (and in this example `annual_spend` would dominate).
It's best to normalise the individual metrics so the average distance accross all pairs of items is 1, so they are on the same scale.
Even then you will want to give more weight to some features than others; this is an iterative process where you use domain knowledge to choose how important each feature will be in your clustering.

An alternative approach to a similarity measure is having a *fitness function* of the clustering.
For example if we were most interested in clustering together customers with similar annual spend we could treat it as a regression problem.
Then the quality of our cluster could be the (cross-validated) root mean square error of predicting the mean annual spend accross a cluster for each customer in that cluster.
We could then use [regression trees](https://en.wikipedia.org/wiki/Decision_tree_learning) over the cluster variables to create the clusters.

# Clustering Algorithm

There are lots of clustering algorithms such as [k-medoids](https://en.wikipedia.org/wiki/K-medoids) (the [k-means](https://en.wikipedia.org/wiki/K-means) equivalent for non-Euclidean distances) and [hierarchical clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering), or more exotic ones like [self-organising maps](https://en.wikipedia.org/wiki/Self-organizing_map).

You can try a few different algorithms but this is normally the easy bit; the hard part is evaluating it.

# Evaluation

Clustering is difficult because there's no one way to evaluate a cluster.
You have to think hard about what evidence that this clustering is going to be useful.
In practice you'll generally want to have the one segmentation accross a number of different use cases, and so you want to check it's useful for all of them.

The best way to look at a cluster is to look at the descriptive and clustering features and make sure they make sense and reveal some insights.
It's always useful to have cluster size; generally clusters that are too small are not useful but you might make an exception for a group of your top few customers that contribute most to your revenue (and similarly too large clusters will wash out useful information, but that might be ok for many customers that make small purchases).

The clustering features will *define* your cluster; in this example it would be the groups of states and industries we consider the same.
You should ask the question *does it make sense for these to go together*?

The descriptive features help understand your cluster; it's worth looking at the centre and the spread, if not the whole distribution.
For example you might want to know the typical number of support calls you get from these customers, the most common aquisition channel and the typical spend.
Measures of spread will tell you how tight the clustering is; common examples are the standard deviation, interquartile range or percentage not in most common category.
If the spread is similar to the whole dataset then the cluster isn't telling you anything useful.
It can be useful to just plot the distributions of individual features to get an idea of how it's composed.

# Iteration

It's important to think through the clustering and descriptive features up front.
Then you can evaluate on different measures, with different weights and different clsutering algorithms as much as necessary.
I've found I often want to constrain the clustering using business knowledge and this is much more difficult with some algorithms than others.

Because the evaluation is subjective it makes sense to start with a simpler clustering method and try tuning feature weights or adding constraints until the clusters look useful.
Reducing the iteration time is really useful for creating reasonable clusters; [Shiny](https://shiny.rstudio.com/) or [ipywidgets](https://ipywidgets.readthedocs.io/en/latest/) are handy tools for experimenting interactively.
As you look more at the clusters you'll get clearer about what a *good* clustering looks like and refine the evaluation criteria.

Generally you want the clustering to be easy to explain, so it's worth thinking about how to do this.
Maybe making it easy to find what cluster something is in is good enough.
Other time you may approximate the final clustering model with a more transparent model.

Clustering isn't the best way to do things like customer targeting; there you're better off building a direct predictive model.
But for extracting information, like reporting, clustering can be a useful tool to understand your dataset.