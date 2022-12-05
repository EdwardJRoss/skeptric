---
categories:
- maths
date: '2020-08-14T22:54:12+10:00'
image: /images/sporadic_groups.svg
title: Classifying Finite Groups
---

Groups can be thought of a mathematical realisation of symmetry.
For example the symmetric groups are all possible permutations of n elements.
Or the dihedral groups are the symmetries of a regular polygon.
A questions mathematicians ask is what kinds of groups are there?

One way to tackle this is to try to decompose them.
One way of doing this is a [decomposition series](https://en.wikipedia.org/wiki/Composition_series) of normal subgroups.

$$ 1 = H_0\triangleleft H_1\triangleleft \cdots \triangleleft H_n = G $$

By the [Jordan-Hölder theorem](https://en.wikipedia.org/wiki/Composition_series) the induced simple quotient groups $$ H_j  / H_{j-1} $$ are unique up to permutation of order.
The [Classification of Finite Simple Groups](https://en.wikipedia.org/wiki/Classification_of_finite_simple_groups) lists all the possible groups.
These can all be seen as sorts of symmetries:

* Cyclic groups of prime order are analogouus to prime numbers
* Alternating group of even permutations
* Groups of [Lie Type](https://en.wikipedia.org/wiki/Group_of_Lie_type), which can be seen as symmetries of [Buildings](https://en.wikipedia.org/wiki/Building_(mathematics))
* 27 Exceptional groups which correspond to special types of symmetries

This is just like how any positive integer can be decomposed into prime factors; in fact the abelian finite groups capture exactly this phenomenon.
However in the general case it's not always trivial to *multiply* two groups together, and there are some options.
This is called the [*extension problem*](https://en.wikipedia.org/wiki/Group_extension).
It turns out this problem is really hard to solve in general.

However we do have all the material we need to generate all groups.
In particular with the [universal embedding theorem](https://en.wikipedia.org/wiki/Universal_embedding_theorem) says that all groups are subgroups of a wreath product.

I don't know of any concrete applications of these kinds of classification.
That being said prime numbers (which are isomorphic to the cyclic groups) have applications in cryptography (such as RSA) *because* they are hard to solve and easy to modify.

Howver overall it's better to focus on the groups as they appear.
In your application are you expecting some strange symmetry related to an unusual group.