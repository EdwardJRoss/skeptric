---
categories:
- math
date: '2020-05-21T21:55:35+10:00'
image: /images/set_containment.png
title: Probability Jaccard
---

I [don't like Jaccard index for clustering](/jaccard-containment) because it doesn't work well on sets of different sizes.
Instead I find the concepts from [Association Rule Learning](https://en.wikipedia.org/wiki/Association_rule_learning) (a.k.a market basket analysis) very useful.
It turns out Jaccard Similarity can be written in terms of these concepts so they really are more general.

The main metrics in association rule mining are the *confidence*, which for pairs is just the conditional probability $$ P(B \vert A) = \frac{P(A, B)}{P(A)} $$
There is also the *lift* which is how much more likely than random (from the marginals) the two events are likely to occur together $$ \frac{P(A, B)}{P(A)P(B)} $$.
Finally there is the *support* which is just $$ P(A, B) $$, but I tend to find the count of the intersection is more useful because it indicated how precise the support, lift and confidence may be.


Notice also that we can write the lift in terms of the confidence of the inverse rule:

$$ \frac{P(A,B)}{P(A)P(B)} = \frac{P(A|B)}{P(A)} $$

If we're looking at items similar to another item *A* then the lift is proportional to the confidence of the inverse rule.

The *Jaccard similarity* of two sets is defined by the equation:

$$ \frac{ \lvert A \cap B \rvert }{ \lvert A \cup B \rvert } $$

Using the inclusion-exclusion principle this can be rewritten as:

$$ \frac{ \lvert A \cap B \rvert }{ \lvert A \rvert + \lvert B \rvert - \lvert A \cap B \rvert } $$

Dividing by the intersection gives an estimate of

$$ \frac{ 1 }{ \frac{1}{P(A|B)} + \frac{1}{P(B|A)} - 1 } $$

So the Jaccard index can be written in terms of the confidence of the association rule and its inverse.
I'm not quite sure how to interpret this; it's something close to (but not quite) a harmonic mean.
This symmetry gives less flexibility than using the lift and confidence together.