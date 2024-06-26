---
categories:
- maths
date: '2020-05-13T21:11:17+10:00'
image: /images/beta_function_contour.png
title: Beta Function
---

The [Beta Function](https://en.wikipedia.org/wiki/Beta_function) comes up in the likelihood of the [binomial distribution](/bernoulli-binomial).
Understanding its properties is useful for understanding the binomial distribution.

The beta function is given by $B(a, b) = \int_0^1 p^{a-1}(1-p)^{b-1} \rm{d}p$ for a and b positive.
If you have $N$ flips of a coin of which $k$ turn heads the likelihood is proportional to $p^{k}(1-p)^{N-k}$ for the probability *p* between 0 and 1.
So the beta function can be seen as the normaliser of the likelihood, with $a = k + 1$ and $b = N - k + 1$ (or inversely $k = a - 1$ and $N   =  a + b - 2$).

The integral can be evaluated directly when b is 1: $B(a, 1) = \int_0^1 p^{a-1} = \frac{1}{a}$.

Using [Integration by Substitution](https://en.wikipedia.org/wiki/Integration_by_substitution) of *p* with *1-p* gives $B(a, b) = B(b, a)$.
This makes sense because in the binomial distribution 0 and 1 are just labels, and if we switch the labels we should get the same overall normalisation.

Using [Integration by Parts](https://en.wikipedia.org/wiki/Integration_by_parts) $B(a, b+1) = \int_0^1 p^{a-1}(1-p)^{b} \rm{d}p = \Big[\frac{p^a(1-p)^b}{a}\Big]_0^1 + \frac{b}{a} \int_0^1 p^{a} (1-p)^{b-1}$.
The first term on the right hand side evaluates to 0 for all positive a and b giving $B(a, b+1) = \frac{b}{a} B(a+1, b)$.

This identity can be repeatedly applied to reduce *b* to 1 when it is a positive integer.

$$B(a, m) = \frac{m-1}{a} B(a+1, m-1) = \frac{(m-1)(m-2)\cdots 1}{a(a+1)\cdots (a+m-1)} B(a+m-1, 1)$$

And so this gives:

$$B(a, m) = \frac{(m-1)!}{a(a+1)\cdots (a+m-1)(a+m)}$$

Multiplying both sides by $(a-1)!$ gives:

$$B(a, m) = \frac{(m-1)! (a-1)(a-2)\cdots 1}{(a+m)(a+m-1)\cdots 1}$$

And so when *a* is also and integer this gives a simple form:

$$B(n, m) = \frac{(m-1)! (n-1)!}{(n+m-1)!} = \frac{1}{(n+m-1) {n+m-2 \choose n-1}}$$

Or in terms of the number of flips *N* of a coin and the number of positive results *k*:

$$B(k+1, N-k+1) = \frac{1}{(N+1) {N \choose k}}$$

The factorial can be generalised to all values with a positive real part using the [Gamma Function](https://en.wikipedia.org/wiki/Gamma_function), which can be seen with an appropriate change of variables:

$$B(a, b) = \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}$$

Many other useful features of the beta function can be obtained from this relation and $\Gamma(a+1) = a \Gamma(a)$.
For example $B(a+1, b) = \frac{a}{a+b} B(a, b)$ and $B(a+2, b) = \frac{a(a+1)}{(a+b)(a+b+1)} B(a, b)$.

These will be useful when looking further into Binomials and the Beta distribution.
