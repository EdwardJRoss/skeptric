---
date: 2010-11-18 17:44:50+00:00
image: /images/surface_banane_trompette.svg
title: Local Lie Groups and Hilbert's Fifth Problem
---

Lie Groups are mathematically “very nice” structures – they are **analytic** manifolds (real or complex) with a group structure such that multiplication and inversion are continuous. They are deeply related to infinitesimal symmetries; a group acting on a space can generally be thought of as a group of symmetries (or automorphisms) of some structure [e.g. rotations ($$SO(n)$$) preserve lengths, Möbius transformations map circles and lines in the complex plane into circles and lines], and the analyticity gives a “local” feel to it – if we know the symmetry locally we can extend it analytically (by the exponential map).


Hilbert’s Fifth Problem essentially asked how restrictive just looking at analytic actions is – what if we looked at continuous actions, how many more groups would we get?


**Theorem [Gleason, Montgomery, Zippin]** For a locally compact group $$G$$ the following are equivalent:

1.  $$G$$ is locally Euclidean.
2.  $$G$$ has no small subgroups; i.e. there exists a neighbourhood of the identity that contains no non-trivial subgroups of $$G$$.
3.  $$G$$ is a Lie Group.

<!--more-->


This is very remarkable: Every topological group that is a manifold is a Lie group! That is if the group operation is continuous and forms a topological manifold then there is an analytic structure on the manifold such that the group operation is analytic.


(Note that this is not true at all for non-group manifolds: In dimensions  $$\geq4$$  there exist topological manifolds that admit no smooth structure, let alone an analytic one. Evidently these can not be given a compatible group structure.)


Now if we take a neighbourhood of the identity we can paste an $$n$$-dimensional Lie group onto an $$n$$-dimensional vector space via the manifold structure – which is extremely convenient for calculations: we have explicit coordinates and everything is analytic so we can extend it (unless we hit a pole). Consequently we get the notion of a Local Lie Group: the restriction of a Lie group to some submanifold.


Essentially the definition of a local group is a Hausdorff topological space with a “locally defined” group: the product and inverse are only defined near the identity and the product is associative (where it’s defined). (For a proper definition see the references later in the post).


We say a local group is globalisable if it is the restriction of some topological group.


If a locally Euclidean local group is globablisable then the solution to Hilbert’s Fifth Problem would imply that it is a local Lie group.


It turns out not every local group is globalisable even if we impose an analytic structure (making it a local Lie group) – a very elegant explicit example is given in a paper by Olver [here](http://www.math.umn.edu/~olver/s_/lg.pdf).


Mal’cev (apparently) proved that a local topological group is globalisable if and only if it is globally associative; for a group it is well known that associativity implies that all products with any number of terms is well defined, however for a local group this is not the case (in this sense a local group is something like a higher categorical group). If all finite products of elements are independent of the order (bracketing) of operations we say it is globally associative. This is the trick in Olver’s paper: he explicitly constructs a local Lie group that has all 3-times products well defined (i.e. associative in the usual sense), but products of 4-terms depends on the bracketing.


However close to the identity any local (Lie) group is globally associative: so some restriction of the group is globalisable.


With all this in mind we have to be a bit careful if we try to look at a local version of Hilbert’s fifth problem, but this is what Isaac Goldbring has [done](http://www.math.ucla.edu/~isaac/ltop.pdf).


He proves that each locally Euclidean topological local group has a restriction that is a local Lie group.


So my question is what can local topological groups do that global ones can’t?


The papers I’ve referenced give some hints: The theory of local topological groups is applicable to the theory of [cancellative topological semigroups](http://www.springerlink.com/content/j344427334w25052/) – but I have trouble thinking of what these mean (if a Lie group is about locally symmetries, a cancellative Lie semigroup is about…? A semigroup would be about mappings (non-invertible transformations) but cancellative is a strong restriction) .


Another interesting construction is the compactification of a group (alluded to in Olver’s paper). As I understand it you can add an “infinite group element” to a non-compact group [that may not be associative, invertible or even cancellative] to make it a compact local group. Admittedly these seem like rather nasty structures though – I doubt analysis would be much easier on this compact structure.


Postscript: There is a lot of excellent information on Hilbert’s Fifth Problem on [Terry Tao’s website](http://terrytao.wordpress.com/category/teaching/254a-hilberts-fifth-problem/).