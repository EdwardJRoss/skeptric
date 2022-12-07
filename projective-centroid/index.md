---
categories:
- data
- maths
date: '2020-12-04T18:17:28+11:00'
image: /images/projective_unit.png
title: Centroid for Cosine Similarity
---

[Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) is often used as a similarity measure in machine learning.
Suppose you have a group of points (like a cluster); you want to represent the group by a single point - the centroid.
Then you can talk about how well formed the group is by the average distance of points from the centroid, or compare it to other centroids.
Surprisingly it's not much more complex than finding the geometric centre in euclidean space, if you pick the right coordinate system.

The cosine distance between two n-dimensional vectors is the cosine of the great circle distance of their projections unit (n-1)-sphere.
The cosine distance between two vectors v and w is given by the rather obtuse formula $\frac{v \cdot w}{\left\| v \right\| \left\| w \right\|}$.
A key property is that it doesn't depend on the length of the vectors; if we double v then it cancels out with the norm in the denominator.
So it's not really a distance between vectors, but actually a distance between *rays* (that is lines with a direction; changing the direction negates the cosine distance).
Then on the unit n-sphere the distance between two points *along the sphere* is just the angle between them at the centre of the sphere.

I define the cosine similarity centroid of a set of points as the ray that has maximum average similarity to the points; it's the most similar point you can have.
For simplicity lets project all the points on the unit sphere, except the origin, which is ignored since all points have the same similarity to it.
Then the cosine distance is just the dot product.
Concretely if our points on the unit sphere are $\{p_i\}_{i=1}^{N}$ then the cosine similarity centroid c is the set of points that maximises $s(c) = \frac{1}{N} \sum_{i=1}^{N} p_i \cdot c$ subject to the constraint that c lies on the unit sphere, that is $\left\| c \right\| = 1$.

The maximum can be found using the [method of Lagrange multipliers](https://en.wikipedia.org/wiki/Lagrange_multiplier).
We want to find the extremum of $L(c, \lambda) = \frac{1}{N} \sum_{i=1}^{N} p_i \cdot c + \lambda (1 - c \cdot c)$
The local extrema are where the partial derivatives are 0: 

$$\frac{\partial L}{\partial c_j}(c,\lambda) = \frac{1}{N} \sum_{i=1}^{N} p_{ij}  - 2 \lambda c_j$$

$$\frac{\partial L}{\partial \lambda}(c,\lambda) = 1 - c \cdot c$$

This occurs where $c = \kappa \sum_{i=1}^{N} p_i$ where $\kappa = \frac{-1}{2 \lambda N}$ is some constant, constrained by the normalisation condition that it lies on the unit sphere.
Taking second derivatives shows this is actually the maximum of L.

That is the centroid lies along the line from the origin to the geometric midpoint in cartesian coordinates of the points on the unit sphere.
So if you want to calculate a centroid of a group of points with respect to the dot product, then normalise the vectors and average them.

Note that this method will fail if there's too much symmetry of the points.
For example if they lie on a regular polyhedron centred at the origin then their mean is zero.
That's because there's no loner a unique most similar point.
For example the centroid for the north pole and the south pole could be any point along the equator.
One easy solution for this is to break the symmetry, by moving any one of the points by a small random amount, and then it will have a solution again.