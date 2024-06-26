---
categories:
- maths
- data
date: '2021-02-16T18:45:26+11:00'
draft: true
image: /images/calculus_variations.png
title: Perturbing Solutions
---

When solving a problem where there are potentially many solutions, an effective way to explore the search space is to *perturb* a solution to reach new solutions.
This way you can efficiently explore the whole manifold of solutions.

I recently came across a problem where I needed to fit a curve subject to some constraints.
It was bounded, it's product with another function needed to have a unique maximum in some range and its average had to be a certain value.
These could be written out as a series of constraints, but it seemed that the problem was severely under-constrained.

One way I thought of to solve it was by an exhaustive search.
By cutting up the x and y axis into N and M pieces respectively then there are $M^N$ possible piecewise linear curves, and each could be evaluated against those constraints.
This was computationally infeasible, and required a better search strategy.
There were ways to better parametrise the curve; for example it was decreasing we could look at the step down at each point rather than the values, but some of the constraints were very difficult to directly parameterise.

However a colleague had already found a curve that met all of the constraints.
Intuitively if we change the curve a small amount in a way that preserves all equality constraints, and doesn't break any inequality constraints, we obtain another solution. 
Equality constraints can generally be used to reparameterise the solutions, reducing the search space, and so we just need to check the inequality constraints.
Once you hit an inequality constraint you've reached the edge in that direction and need to go in a different direction.

By coming up with perturbations that were likely to preserve the constraints we were able to explore a considerable part of the solution space and come up with alternative solutions that may be better for other reasons.