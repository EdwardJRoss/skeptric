---
categories:
- maths
- data
date: '2021-03-03T08:00:00+11:00'
image: /images/exponential_power_distributions.png
title: Probability Distributions Between the Mean and the Median
---

The normal distribution is used throughout statistics, because of the [Central Limit Theorem](https://en.wikipedia.org/wiki/Central_limit_theorem) it occurs in many applications, but also because it's computationally convenient.
The expectation value of the normal distribution is the mean, which has many nice arithmetic properties, but the drawback of being sensitive to outliers.
When discussing [constant models](/constant-models) I noted that the minimiser of the [Lᵖ error](https://en.wikipedia.org/wiki/Lp_space#The_p-norm_in_finite_dimensions) is a generalisation of the mean; for $$ p = 2 $$ it's the mean, for $$ p = 1 $$ it's the median, and for $$ p = \infty $$ it's the midrange (half way betwen the maximum and minimum points).
We can similarly generalise the normal distribution to a family of exponential distributions, $$ \frac{A}{\sigma} e^{- \alpha \left\vert\frac{x - \mu}{\sigma}\right\vert^p} $$ where the expectation value $$\mu$$ is the minimiser of the Lᵖ error.
This distribution can then be used for estimates less sensitive to outliers, for p between 1 and 2.

## Calculating the moments

Consider the probability distributions of the form above: 

$$ f(x) = \frac{A}{\sigma} e^{- \alpha \left\vert\frac{x - \mu}{\sigma}\right\vert^p} $$

The constant A is determined by normalisation $$ \int_{-\infty}^{\infty} f(x) = 1 $$.
Using the [power exponential formulas](/integrating-power-exponential) gives the normalising constant as $$ A = \frac{p \alpha^{\frac{1}{p}}}{2 \Gamma\left(\frac{1}{p}\right)} $$.
Using normalisation and symmetry the mean is $$ \int_{-\infty}^{\infty} x f(x) = \mu $$.
With a little more caluclation using the power exponential formulas it can be shown the variance is $$ \int_{-\infty}^{\infty} x^2 f(x) =  \sigma^2 $$ if $$ \alpha = \left(\frac{\Gamma\left(3/p\right)}{\Gamma\left(1/p\right)}\right)^{p/2} $$.
So for example in the case $$ p = 2 $$ we get the familiar constant of 1/2 in the normal distribution exponent.

Given data how do we estimate the mean $$ \mu $$ and the standard deviation $$ \sigma $$ for the distribution $$\frac{A}{\sigma} e^{- \alpha \left\vert\frac{x - \mu}{\sigma}\right\vert^p} $$?
With the data $$ X_1, X_2, \ldots, X_n $$ the log-likelihood is:

$$ l(\mu, \sigma) = N \log(A) - N \log(\sigma) - \alpha \sum_{i=1}^{N} \left\vert \frac{X_i - \mu}{\sigma} \right\vert ^ p $$

The Maxmium Likelihood Estimator for the expectation value of the distribution $$ \mu $$ is the minimiser of $$\left\vert X_i - \mu \right\vert^ p$$, which is precisely the minimiser of the Lᵖ error.
So for $$ p = 2 $$ the expectation value is estimated by the mean, for $$ p = 1 $$ by the median and as $$ p \rightarrow \infty $$ by the midrange.
The Maximum Likelihood Estimator for $$ \sigma $$ is

$$ \hat{\sigma} = \left(\frac{p \alpha}{N} \sum_{i=1}^{N} \left\vert X_i - \mu \right\vert^p\right)^{1/p} $$

which is, up to a multiplicative constant, the Lᵖ error.

So by modelling data under the probability distribution $$ \frac{A}{\sigma} e^{- \alpha \left\vert\frac{x - \mu}{\sigma}\right\vert^p} $$ the Maximum Likelihood Estimate for $$ \mu $$ is the minimiser of Lᵖ error, and for $$ \sigma $$ is the Lᵖ error up to a constant factor.
This could be used as a model in a regression allowing different kinds of estimates; in particular for $$p = 1$$, with distribution $$ \sqrt{\frac{3}{2}} e^{-\sqrt{6}\left\vert\frac{x - \mu}{\sigma}\right\vert} $$ the MLE estimate is the median and the MAD is $$ \frac{\hat{\sigma}}{\sqrt{6}} $$.
A reparameterisation of the exponent variable could ensure the MLE is the Lᵖ error, but then the variable would no longer correspond with the square root of the variance of the distribution.