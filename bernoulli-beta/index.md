---
categories:
- ''
date: '2020-05-12T23:26:18+10:00'
draft: true
image: ''
title: Bernoulli Beta
---

Suppose 3 of 10 people that hit a landing page follow the call to action, how likely is it that the conversion rate is 30%?
What if it were 30 of 100 people?
In the [previous article](/bernoulli-binomial) we looked at how the binomial distribution told us if we know the conversion rate the probability that any given number of people would follow the call to action.
We build on that to solve the inverse problem: estimating the conversion rate from the data.

Given any conversion rate *p* the probability of getting *k* successes from *N* people is $$ P(Z=k \vert P=p) = {N \choose k} p^k (1-p)^{N-k} $$.
If we consider this probability as a function of *p* it's called the *likelihood function*.
Let's assume that the conversion rate, *P*, is equally likely to be any number between 0 and 1.
Then it turns out that the best estimate for the probability is proportional to the likelihood.

That is $$ P(P=p \vert Z=k) = \frac{p^k (1-p)^{N-k}}{B} $$, where *B* is some constant so the integral over *p* over the whole range from 0 to 1 is 1, to make it an actual probability.
By finding where the derivative with respect to *p* is zero we get the *maximum likelihood estimate*, which is $$ \frac{k}{N} $$.
That is the most likely value for the probability for success is the fraction of successes, which is intuitive.

However there's a lot more to the distribution than the most likely value.
This distribution gives an estimate for the range of

This is in fact a special case of the [Beta Distribution](https://en.wikipedia.org/wiki/Beta_distribution); $$ \beta(p; a, b) = \frac{p^{a-1} (1-p)^{b-1}}{B(a, b)} $$, where *B* is a normalising constant equal to the integral of the numerator from 0 to 1.
When *a* and *b* are both 1, then it is a flat distribution between 0 and 1 (so any value is equally likely).
In our example above we had $$ \beta(p; k+1, N-k+1) $$.