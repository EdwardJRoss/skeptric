---
categories:
- data
date: '2020-05-25T20:53:28+10:00'
image: /images/cluster.svg
title: Clustering for Exploration
---

Suppose you're running a website with tens of thousands of different products, and no satisfactory way to group them up.
Even a mediocre clustering can really help bootstrap your understanding.
You can use the clusters to see new patterns in the data, and you can manually refine the clusters much more easily than you can make them.

There are many techniques to [cluster structured data](/clustering-segmentation) or even detect them as [communities](/community-detection) in [the graph of interactions with your users](/recommendation-graph).
The problem is evaluating the clusters is very difficult, and requires a lot of product expertise.
There are in fact different useful ways to group products; for example a retailer could group together all books, or they could group the Harry Potter books with the Harry Potter movies.
Depending on your application you could use a different grouping; in fact you could have overlapping hierarchies, or just a generic measure of difference, but they're harder to use.

Trying to group things manually even with thousands of products is hard; the number of possible groupings grows exponentially with the number of items.
However if you have an example grouping, splitting the thousands of products into a couple dozen groups, it's straightforward to manually check if it makes sense.
This can be a great tool for better understanding the products and how they relate, and it's relatively easy to move items between groups manually to improve the grouping.
This means using any technique to get a cluster in the right ballpark will be very useful for exploring the data in ways that would be very difficult without it.

This kind of balance between using algorithms and domain expertise is very powerful; while it's not totally data driven you can use knowledge not in the data (for example on new products) to improve the result.
