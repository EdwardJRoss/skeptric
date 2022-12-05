---
categories:
- data
- bayes
date: '2021-09-20T13:53:33+10:00'
image: /images/beta_distribution.png
title: Bernoulli Trials and the Beta Distribution
---

## Estimating a Bernoulli Probability

Suppose we want to know the probability of an event occurring; it could be a customer converting, a person contracting a disease or a student passing a test.
This can be represented by a Bernoulli Distribution, where each draw is an independent random variable $$ \gamma_i \sim {\rm Bernoulli}(\theta) $$.
The only possible values are failure (represented by 0) with probability $$\theta$$ and success (represented by 1) with probability $$1-\theta$$ (although the labels are completely arbitrary and we can switch them by setting $$\eta_i = 1 - \gamma_i$$ then $$\eta_i \sim {\rm Bernoulli}(1-\theta) $$).

The probability distribution can be conveniently written as $$ {\mathbb P}(\gamma = k) = \theta^{k}(1-\theta)^{1-k} $$, since $$ {\mathbb P}(\gamma = 1) = \theta^{1}(1-\theta)^{0} = \theta$$ and $${\mathbb P}(\gamma = 0) = \theta^{0}(1-\theta)^{1} = 1 - \theta$$.
This form is convenient because for multiple variables the probabilities multiply (since the variables are independent), and the exponents add, giving a simple expression.
In particular 

$$\begin{align}
{\mathbb P}(\gamma_1=k_1,\ldots,\gamma_N=k_N) &= {\mathbb P}(\gamma_1=1) \cdots {\mathbb P}(\gamma_N=k_N) \\
&= \theta^{k_1 + \cdots + k_N} (1 - \theta)^{N - (k_1 + \ldots k_N)} \\
&= \theta^{z}(1-\theta)^{N-z}
\end{align} $$

where z is the number of positive results (which is as in the [binomial distribution](/bernoulli-binomial), up to multiplicity from different orderings).
Note that the result just depends on the total number of trials and the number of successes.

In the Bayesian framework the posterior probability distribution of $$\theta$$ can be estimated conditional on the observed data; in particular from Bayes rule:

$$\begin{align}
{\mathbb P}\left(\theta \vert \gamma_1=k_1, \ldots, \gamma_N=k_N\right) &= \frac{{\mathbb P}\left(\theta \vert \gamma_1=k_1,\ldots,\gamma_N=k_N\right)P(\theta)}{P(\gamma_1=k_1,\ldots,\gamma_N=k_N)} \\
&= \frac{\theta^z(1-\theta)^{N-z}P(\theta)}{\int_0^1 P(\gamma_1=k_1,\ldots,\gamma_k \vert \theta=k_N) P(\theta) \,{\rm d}\theta}
\end{align}$$

To get a posterior distribution we need to choose an appropriate prior.
A *flat prior* is a reasonable starting point if we know nothing about the situation, $$P(\theta) = 1, \; \forall \theta \in[0,1]$$.
Then from the above the posterior will be proportional to $$\theta^{z}(1-\theta)^{N-z}$$ (up to a normalising constant).
This is a special case of the Beta distribution; if $$ \Theta \sim {\rm Beta}(\alpha,\beta) $$ for positive $$\alpha, \beta$$ then

$$ P(\Theta=\theta) = {\rm Beta}(\alpha, \beta)(\theta) = \frac{\theta^{\alpha-1}(1-\theta)^{\beta-1}}{B(\alpha, \beta)} $$

Where the normalising denominator is the Beta Function $$B(\alpha, \beta) = \int_{0}^{1} \theta^{\alpha}(1-\theta)^{\beta-1}\, {\rm d}\theta = \frac{\Gamma(\alpha)\Gamma(\beta)}{\Gamma(\alpha + \beta)}$$.
Notice that in particular $${\rm Beta}(1,1)$$ is the (flat) uniform distribution on [0,1].

The special thing about the Beta Distribution is it's a *conjugate prior* for Bernoulli trials; with a Beta Prior distribution for the probability of positive cases $$\theta$$ then the posterior is also a Beta distribution.
Specifically $${\mathbb P}\left(\theta \vert \gamma_1=k_1, \ldots, \gamma_N=k_N\right) \propto \theta^{k + \alpha - 1}(1 - \theta)^{N-k + \beta - 1}$$, and so the posterior is distributed as $${\rm Beta}(\alpha + z, \beta + N-z)$$, and in particular for a uniform prior it is $${\rm Beta}(z + 1, N-z+1)$$.

## Properties of the Beta Distribution

Since given a flat (or more generally Beta) prior we get a Beta posterior for the Bernoulli probability $$\theta$$ it makes sense to study the properties of the Beta distribution to understand $$\theta$$.

### Maximum likelihood

The most likely value can be found with a bit of differential calculus.
The derivative is

$$\frac{{\rm d}{\rm Beta}(\alpha, \beta)}{{\rm d} \theta}(\theta) =  \frac{\theta^{\alpha-2}(1-\theta)^{\beta-2}}{B(\alpha, \beta)}\left(\alpha - 1 - (\alpha + \beta - 2)\theta\right)$$

which may be zero at $$\hat{\theta} = 0, 1, \frac{\alpha - 1}{\alpha + \beta - 2}$$.
The extremum is a maximum when the second derivative is negative.
The second derivative at the local extrema are:

$$\begin{align}
\frac{{\rm d^2}{\rm Beta}(\alpha, \beta)}{{\rm d} \theta^2}(\hat{\theta}) &= \frac{\rm d}{{\rm d}\theta}\left.\left(\frac{\theta^{\alpha-2}(1-\theta)^{\beta-2}}{B(\alpha, \beta)}\right)\right\vert_{\theta=\hat\theta}\left((\alpha - 1)- (\alpha + \beta - 2)\hat\theta\right) \\
&- \left(\frac{\hat\theta^{\alpha-2}(1-\hat\theta)^{\beta-2}}{B(\alpha, \beta)}\right) (\alpha + \beta - 2)\\
 &=- \left(\frac{\hat\theta^{\alpha-2}(1-\hat\theta)^{\beta-2}}{B(\alpha, \beta)}\right) (\alpha + \beta - 2)\\
 &=- {\rm Beta}(\alpha, \beta)(\hat\theta) \frac{(\alpha + \beta - 2)}{\hat\theta(1-\hat\theta)}
\end{align}$$.

which is negative if and only if $$0 < \hat\theta < 1 $$ and $$\alpha + \beta > 2$$.
In the case when $$\alpha = 1$$ and $$\beta > 1$$ then the derivative $$-\frac{\theta^{-1}(1-\theta)^{\beta-2}}{B(1, \beta)}(\beta - 1)\theta$$ is negative on the whole interval (0,1), and so the function decreases from its maximum value at $$\hat\theta=0$$.
Similarly when $$\beta=1$$ and $$\alpha > 1$$ then $$\hat\theta=1$$ and the derivative is positive on the whole interval (0,1) and so the function increases to its maximum valu at $$\hat\theta=1$$.
So overall in all cases where $$\alpha + \beta > 2$$ the maximum likelihood occurs at $$\hat\theta=\frac{\alpha - 1}{\alpha + \beta - 2}$$ which is necessarily in the interval [0,1].

This gives the same results as a Maximum Likelihood analysis for the Binomial when we observe z successes from N trials with a uniform prior.
With the $${\rm Beta}(z+1, N-z+1)$$ distribution the maximum likelihood estimator is $$\hat\theta=\frac{z}{N}$$, the proportion of successes.
The second derivative also matches an approximate normal distribution of $$\theta$$ with standard deviation $$\sqrt{\frac{\hat\theta(1-\hat\theta)}{N}}$$ as would be obtained from the [efficiency of Maximum Likelihood Estimators](https://en.wikipedia.org/wiki/Maximum_likelihood_estimation#Efficiency) using the [Fisher Information Matrix](https://en.wikipedia.org/wiki/Fisher_information) for the binomial.

When $$\alpha + \beta \leq 2$$ there isn't necessarily a maximum likelihood estimate.
When $$\alpha = \beta = 1$$ then all values are equally likely.
The [Jeffrey's Prior](https://en.wikipedia.org/wiki/Jeffreys_prior#Bernoulli_trial) is $${\mathbb P}(\theta) \propto \sqrt{I(\theta)} = \frac{1}{\sqrt{\theta(1-\theta)}}$$ and so corresponds to a $${\rm Beta}(1/2,1/2)$$.
In this case there is a *local minimum* at 1/2, and the most likely values are 0 and 1.
But as soon as we add any data to a Jeffrey's prior we do have a most likely estimate; for $$\alpha \leq 1$$ and $$\alpha + \beta \geq 2$$ then the derivative is negative on the whole interval (0,1) and so the probability decreases from its maximum value at 0.
Similarly for $$\beta \leq 1$$ and $$\alpha + \beta \geq 2$$ then the derivative is positive on the whole interval (0,1) and the probability increases to its maximum value at 1.

The Bayesian framework allows us to ask questions that are harder just using asymptotic analysis.
For example we can calculate things like how likely $$\theta$$ is greater than 1/2, and come up with a credible interval for the parameter based on the data.
The cost of this is having to specify a prior (and some extra calculations).

### Mean and variance

The mean can be calculated using the [properties of the Beta function](/beta-function).
Given $$\Theta \sim {\rm Beta}(\alpha, \beta)$$ then 

$$\begin{align}
{\mathbb E}(\Theta) &= \int_0^1 \theta {\mathbb P}(\Theta=\theta) \, {\rm d}\theta\\
&= \int_0^1 \frac{\theta^{\alpha}(1-\theta)^{\beta-1}}{B(\alpha, \beta)} \\
&= \frac{B(\alpha + 1, \beta)}{B(\alpha, \beta)} \\
& = \frac{\alpha}{\alpha + \beta}
\end{align} $$

Notice that the mean value is well defined for all positive $$\alpha, \beta$$, and when the mode exists it is closer to the edges of the distribution than the mean.

We can similarly calculate the expectation of the square:


$$\begin{align}
{\mathbb E}(\Theta^2) &= \int_0^1 \theta^2 {\mathbb P}(\Theta=\theta) \, {\rm d}\theta\\
&= \frac{B(\alpha + 2, \beta)}{B(\alpha, \beta)} \\
&= \frac{\alpha(\alpha+1)}{(\alpha+\beta+1)(\alpha+\beta)}
\end{align} $$


This then gives variance 

$$\begin{align}
{\mathbb V}(\Theta) &= {\mathbb E}\left((\Theta - {\mathbb E}(\Theta))^2\right)\\
&={\mathbb E}(\Theta^2) - {\mathbb E}(\Theta)^2 \\
&= \frac{\alpha}{\alpha+\beta}\left(\frac{\alpha+1}{\alpha+\beta+1} - \frac{\alpha}{\alpha+\beta}\right) \\
& = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}
\end{align}$$

## Parameterisations of Beta Distribution

Summarising our previous results we have for a $${\rm Beta}(\alpha, \beta)$$ distribution the mean is $$\mu = \frac{\alpha}{\alpha + \beta}$$, the variance is $$\sigma^2 = \frac{\alpha\beta}{(\alpha+\beta)^2(\alpha+\beta+1)}$$ and the mode, for $$\alpha, \beta \geq 1$$ and $$\alpha + \beta > 2 $$ is $$\omega = \frac{\alpha -1}{\alpha+\beta-2}$$.
However we can use these properties to themselves define the Beta distribution which is useful for different contexts.

Firstly note that $$\alpha$$ is analogous to the number of successes and $$\beta$$ is analogous to the number of failures in the Bernoulli trials.
These are additive, so that given an $${\rm Beta}(\alpha, \beta)$$ prior and z successes with $$v = N-z$$ failures the posterior is $${\rm Beta}(\alpha + z, \beta + v)$$.
So in this parameterisation the successes and failures add.

Another way to look at it is in terms of size $$\kappa = \alpha + \beta$$ and the mode $$\omega$$.
These are analogous to the number of trials and proportion of successes respectively.
We can rewrite $$\alpha = (\kappa - 2)\omega + 1$$ and $$\beta = (\kappa - 2)(1- \omega) + 1$$, and for $$\kappa \geq 2$$ we can always express the Beta function in terms of $$\kappa$$ and $$\omega$$.
Given N trials with a proportion of successes $$p=z/N$$, the posterior has size $$\kappa' = \kappa + N$$, and posterior mode $$\omega' = \frac{(\kappa - 2)}{N + \kappa - 2} \omega + \frac{N}{N + \kappa - 2} p $$, so it's a weighted average of the individual probabilities.
In summary sample sizes add, and proportions combine as a weighted average, which makes intuitive sense when thinking about combining the results of Bernoulli trials.

Finally sometimes it can be useful to think in terms of the mean and the variance.
These don't have quite as clean as an interpretation in terms of the data, but the mean represents how skewed the data is (in a slightly less extreme way than the mode), and the variance is inversely related to the size since the certainty increases with more data.
The size can be expressed as $$\kappa = \frac{\mu(1-\mu)}{\sigma^2} - 1$$ and $$ \alpha = \kappa \mu $$, $$\beta = \kappa(1-\mu)$$.
They combine in a more complex way.

## Summary

For binomial trials the Beta distribution occurs naturally as a conjugate prior for the binomial probability $$\theta$$.
Starting with a uniform prior and adding data with N trials and z successes we get a $${\rm Beta}(z+1,N+z-1)$$ posterior for $$\theta$$.
This has its maximum probability at the sample proportion $$p=z/N$$, and we can alternately write the distribution as $${\rm Beta}(Np + 1, N(1-p) + 1)$$.
The sample proportions combine as a weighted average; given $$N_1, N_2$$ trials with sample proportions $$p_1, p_2$$ the combined size is $$N_1 + N_2$$ with proportion $$\frac{N_1 p_1 + N_2 p_2}{N_1 + N_2}$$.

Choosing a $${\rm Beta}(\alpha, \beta)$$ prior is equivalent to starting with a flat prior and adding an additional $$\alpha - 1$$ successes and $$\beta - 1$$ failures; or equivalently having a successful proportion of $$\omega = \frac{\alpha-1}{\alpha + \beta - 2}$$ out of $$\kappa - 2 = \alpha + \beta - 2$$ trials.
This framing is useful in understanding hierarchical binomial models.
A lot of this is based heavily on Chapter 6 of Kruschke's [Doing Bayesian Data Analysis](http://doingbayesiandataanalysis.blogspot.com/).