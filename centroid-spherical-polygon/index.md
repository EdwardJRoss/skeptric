---
categories:
- maths
date: '2023-06-07T08:00:00+11:00'
image: /images/great_circle_distance.svg
title: Centroid Spherical Polygon
---

You're organising a conference of operations research analysts from all over the world, but their time is very valuable and they only agree to meet if you make the travel distance fair, even if they have it on a boat in the middle of the ocean.
Where do you put the conference?

First we need to agree on what is the fairest, and a good candidate is average squared distance.
You could minimise the average distance travelled, which would likely be best for the environment, but it could result in some people travelling a long way and other people travelling nowhere.
An alternative is to minimise the average *square* distance travelled; squared distances penalise outliers more heavily so it reduces the number of people who have to travel a really long way.
If the people all lived on a straight line minimising the average distance would be the median, and minimising square distance would be the mean (as these are the best [constant models](/constant-models) under these metrics); you could even minimise [some other statistic](distribution-between-mean-median) such as some power of the distance.
If the people live close enough that we can ignore the curvature of the Earth the meeting place would be the centroid of their polygon (the mean of their coordinates), which seems intuitively reasonable.

Let's model the world as a unit sphere in 3 dimensional space, and have the N people at cartesian coordinates $\{ p_i\}_{i=1}^{N}$.
Then the point c of minimum average squared distance, the spherical centroid or [spherical Fréchet mean](https://en.wikipedia.org/wiki/Fr%C3%A9chet_mean), is given by the formula:

$$c = k \sum_{i=1}^{N} \frac{\arccos\left(c \cdot p_i \right)}{\sqrt{1 - (c \cdot p_i)^2}} p_i$$

where k is a normalising constant so that c lies on the unit sphere.
This is like the mean of the points, but weighted based on the similarity of the centroid to those points.

The equation doesn't solve for c since it's on both sides of the equation, but in some cases can be evaluated iteratively.
If you start with a random point $c$ on the sphere you can update the estimate until it converges.
In highly symmetric and degenerate cases this will fail to find the optimum, or cycle between points, but in many cases it will work.

## Context

I found a few discussions of this problem but no actual solution.
This question was inspired by a [Notion Parrallax blog post](https://notionparallax.co.uk/2009/centroid-of-points-on-the-surface-of-a-sphere).
In there he suggests a few methods but he doesn't actually solve it.
There's a GIS Stackexchange question on [Calculating a Spherical Polygon Centroid](https://gis.stackexchange.com/questions/43505/calculating-a-spherical-polygon-centroid).
[One of the answers](https://gis.stackexchange.com/a/44767) references the paper [Spherical Averages and Applications to Spherical Splines and Interpolation, by Buss and Fillmore, *ACM Transactions on Graphics* 20, 95126 (2001)](http://math.ucsd.edu/~sbuss/ResearchWeb/spheremean/).
However the paper gives a really complex derivation of what is essentially gradient descent (and they forget to multiply by a small step size) and I found it hard to follow.
But this is very similar to [finding the centroid of cosine similarity](/projective-centroid) and we can take the same approach.

## Solution

The extrema of average distance can be found using the [method of Lagrange multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier).
The average square distance to a point c on the sphere is given by $d(c) = \frac{1}{N} \sum_{i=1}^{N} \arccos^2\left( c \cdot p_i \right)$.
Then the lagrangian that constrains c to the unit sphere is given by $L(c, \lambda) = \sum_{i=1}^{N} \arccos^2\left( c \cdot p_i \right) + \lambda (1 - c \cdot c)$.

Taking the partial derivatives (using [arccos derivative](https://math.berkeley.edu/~peyam/Math1AFa10/Arccos.pdf)) we get (where $p_{ij}$ is a clumsy notation meaning the jth coordinate of the ith point):

$$\frac{\partial L}{\partial C_j}(c, \lambda) = - \sum_{i=1}^{N} \frac{2 \arccos(c \cdot p_i) p_{ij}}{\sqrt{1 - (c \cdot p_i)^2}} - 2 \lambda c_j$$

$$\frac{\partial L}{\partial \lambda}(c, \lambda) = 1 - c \cdot c$$

The extrema occur where these are 0, and in particular where $c = k \sum_{i=1}^{N} \frac{p_i}{\sqrt{1 - (c \cdot p_i)^2}}$ where $k=\frac{1}{\lambda}$ is a constant so that the normalisation constraint holds $c \cdot c = 1$.

What's not obvious to me is when this point will be the minimum, and when calculating it iteratively will converge.
In some numerical experiments I've found when there's an even number of points there are often multiple minima and the solution will cycle between them.
It would be useful to have some criteria under which this converges.

[Slava Andrejev](https://www.linkedin.com/in/slava-andrejev/), who provided corrections to this article, notes that the formula is well defined even where $c = p_i$, and in fact the second order Taylor expansion of the weight is $\frac{\arccos(x)}{\sqrt{1-x^2}} \approx (22 + x (2 x - 9)) / 15 + O((1-x)^3)$ is very good near $x=1$.


## Minimum Distance Solution

I originally made the mistake of calculating the minimum distance solution, which is analogous to the median (thanks to Slava Andrejev for pointing out this mistake).
This is also a kind of Fréchet mean, but with the metric from the square root of the geodesic distance rather than the geodesic distance, so not a traditional centroid.
In this case we want to minimise $d(c) = \frac{1}{N} \sum_{i=1}^{N} \arccos\left( c \cdot p_i \right)$, and using the same argument we get the minimum at:

$$c = k \sum_{i=1}^{N} \frac{1}{\sqrt{1 - (c \cdot p_i)^2}} p_i$$

where k is a normalising constant so that c lies on the unit sphere.
The weighting does not have the $\arccos$ term and so it's undefined for $c = p_i$.
Still in many experiments for a good $c$ this will converge (and often converges for $c=0$).
