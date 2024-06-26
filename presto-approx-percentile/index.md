---
categories:
- presto
- athena
date: '2020-09-12T08:00:00+10:00'
image: /images/qdigest_layout.png
title: Approximate Percentiles in Presto and Athena
---

Calculating percentiles and quantiles is a common operation in analytics.
While they can be done in vanilla SQL with window functions and row counting, it's a bit of work and can be slow and in the worst case can hit database memory or execution time limits.
Presto (and Amazon's hosted version Athena) provide an [approx_percentile](https://prestodb.io/docs/current/functions/aggregate.html#approx_percentile) function that can calculate percentiles approximately on massive datasets efficiently.

When running this I found that it was non-deterministic.
This is really annoying because it makes testing hard, especially diff testing running on a production dataset.
I wanted understand why; I think the reason is that the approximation depends a little on the order items are inserted.
Presto runs across multiple nodes, so it's likely for the data to come in different orders depending on hardware details (though this can even happen on a single node if the way it fetches the data blocks is non-deterministic).

The rest of this post documents a bit about how it works.

From the documentation:

>  As approx_percentile(x, percentage), but with a maximum rank error of accuracy. The value of accuracy must be between zero and one (exclusive) and must be constant for all input rows. Note that a lower “accuracy” is really a lower error threshold, and thus more accurate. The default accuracy is 0.01.

The implementation is in the Approximate Percentile Aggregation functions in the [aggregations](https://github.com/prestodb/presto/tree/master/presto-main/src/main/java/com/facebook/presto/operator/aggregation).
In particular most of them build on [ApproximateLongPercentileAggregations](presto-main/src/main/java/com/facebook/presto/operator/aggregation/ApproximateLongPercentileAggregations.java).
In turn this uses `com.facebook.airlift.stats.QuantileDigest` to do a lot of the heavy lifting.

[Airlift](https://github.com/airlift/airlift) describes itself as:

> Airlift is a framework for building REST services in Java.
>
> This project is used as the foundation for distributed systems like Presto.

This seems like a strange place to do the heavy lifting of quantiles, but apparently it is.
It has implementation of Quantile Digest for approximate percentiles, and Hyperloglog.

They've got a [description](https://github.com/airlift/airlift/blob/master/stats/docs/qdigest.md) of how they calculate it, and there are a lot of useful comments in the [source](https://github.com/airlift/airlift/blob/master/stats/src/main/java/io/airlift/stats/QuantileDigest.java).


> Implements http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.132.7343, a data structure
> for approximating quantiles by trading off error with memory requirements.
> 
> The size of the digest is adjusted dynamically to achieve the error bound and requires
> O(log2(U) / maxError) space, where *U* is the number of bits needed to represent the
> domain of the values added to the digest. The error is defined as the discrepancy between the
> real rank of the value returned in a quantile query and the rank corresponding to the queried
> quantile.
> 
> Thus, for a query for quantile *q* that returns value *v*, the error is
> |rank(v) - q * N| / N, where N is the number of elements added to the digest and rank(v) is the
> real rank of *v*
> 
> This class also supports exponential decay. The implementation is based on the ideas laid out
> in http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.159.3978

In the source code it mentions:

> Get an upper bound on the quantiles for the given proportions. A returned q quantile is guaranteed to be within
> the q and q + maxError quantiles.

This is a little interesting; following the code calls I think `approx_percentile` in Presto will tend to slightly overestimate the percentile and never underestimate.

This follows from the paper they implemented: [Medians and Beyond: New Aggregation Techniques for Sensor Networks](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.132.7343).

> This error is bounded by εn (Theorem 1).  So, the rank of value reported by our algorithm is between qn and (q+ε)n. Thus the error in our estimate is always positive, i.e., we always give a value which has a rank greater than (or equal to) the actualquantile

Skimming the paper they build a binary tree, called a *q-digest*, to estimate the quantiles.
This makes intuitive sense; you only need bounds of regions when calculating quantiles, but to get those bounds accurately you need lots of memory (though I'm sure there are lots of messy details).

However I suspect that the output tree depends on the order of the input.
Then thinking about how Presto uses it; the order items are inserted into it won't always be the same.
So this is likely why it's non-deterministic.