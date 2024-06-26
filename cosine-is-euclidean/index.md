---
categories:
- maths
- data
date: '2020-12-14T21:06:03+11:00'
image: /images/law_of_cosines_acute.svg
title: Cosine Similarity is Euclidean Distance
---

In mathematics it's surprising how often something that's obvious (or *trivial*) to someone else can be revolutionary (or weeks of work) to someone else.
I was looking at the [annoy (Approximate Nearest Neighbours, Oh Yeah)](https://github.com/spotify/annoy) library and saw this comment:

> Cosine distance is equivalent to Euclidean distance of normalized vectors 

I hadn't realised it at all, but once the claim was made I could immediately verify it.
Given two vectors u and v their distance is given by the length of the vector between them: $d = \| u - v \| = \sqrt{(u - v) \cdot (u - v)}$.
Expanding this out and using $u \cdot v = \|u\|\|v\| \cos \theta$, where $\theta$ is the angle between the two vectors, gives

$$d = \sqrt{\|u\|^2 + \|v\|^2 - 2 \|u\|\|v\| \cos \theta }$$

For unit vectors the norms are 1 and this reduces to

$$d = \sqrt{2 (1 - cos \theta)}$$

So cosine similarity is closely related to Euclidean distance.
Of course if we used a sphere of different positive radius we would get the same result with a different normalising constant.
Thus $\sqrt{1 - cos \theta}$ is a distance on the space of rays (that is directed lines) through the origin.

The centroid for cosine similarity is easy to calculate; project the points on some sphere, calculate their Euclidean centroid (that is average them) and take the ray through that point.
I [proved this using Lagrange multipliers](/projective-centroid), where I defined the centroid as the point that maximises average cosine similarity; this is the same as minimising the average Euclidean distance and so it really is a centroid.
A plausible way to see this is to note that the Euclidean centroid is the distance minimiser in Euclidean space, and the projection to the sphere is the closest point on the sphere to the centroid, so this projection must be the centroid for cosine similarity.