---
categories:
- maths
- data
date: '2021-10-12T12:39:39+11:00'
image: /images/james_stein_shrinking.png
title: Estimating Group Means with Empirical Bayes
---

When calculating the averages of lots of different groups it doesn't make sense to treat the groups as independent, but to *pool information* across groups, especially on groups with little data.
One way to do this is to build a model on covariates of the groups or on [categorical embeddings](/categorical-embeddings) to use information from other observations to inform that observation.
Surprisingly [Stein's Paradox](/stein-paradox) says if we're trying to minimise the root mean square error, have only one observation per group, and they're all normally distributed with the same standard deviation we're always better off shrinking the means towards a common point than treating them separately (even if they're completely unrelated quantities!).
Many of these conditions can be weakened, and in practice when estimating averages of related quantities the gain from shrinking (even when we don't have covariates) could be substantial.

Calculating averages (or proportions) in groups is a very common procedure for analysts, which is why `GROUP BY` is an essential feature of SQL.
Some examples are calculating the average sales by salesperson, the conversion by referrer, [salary by job ad title](/job-ad-title-salary), the success rate of a surgery by surgeon, [hit-rates of baseball players](http://doingbayesiandataanalysis.blogspot.com/2012/11/shrinkage-in-multi-level-hierarchical.html), average runs of cricket players, or student test scores by classroom.
The ordinary procedure is to treat each group independently, and sum and divide, but this can lead to very unstable estimates for small groups.
An extreme example is calculating a proportion with only 1 observation which is necessarily 0% or 100% and so always will be at the top or the bottom.
A common approach to these is to filter out groups with too few observations, and a less common approach is to use [confidence intervals](https://www.evanmiller.org/how-not-to-sort-by-average-rating.html) to represent the uncertainty.
A less common approach is to use information from other groups to inform the estimate for that group; but is used and known by names like random effects models (or more generally mixed-effects models), multilevel models or hierarchical models (among others).

# Empirical Bayesian Model

This can be easily motivated by a Bayesian approach, following chapters 6 and 7 of [Computer Age of Statistical Inference](https://web.stanford.edu/~hastie/CASI/).
Assume that the observations in each group are normally distributed with an unknown mean, $\theta_i$ for the ith group, and known standard deviation $\sigma_i$ (more on how we could know this later), so a random draw X from group i satisfies $X \sim {\mathcal N}(\theta_i, \sigma_i^2)$.
For the sake of argument assume we've got one observation from each group; in general if we've got several observations $X_{i;1}, \ldots, X_{i;N_i} \sim {\mathcal N}(\theta_i, \sigma_i^2)$ we can replace it with their average $X_i := \frac{1}{N_i} \sum_{j=1}^{N_i} X_{i;j} \sim {\mathcal N}(\theta_i, \sigma_i^2/N_i)$.
If we've got a binomial proportion we can use the [normal approximation](https://en.wikipedia.org/wiki/Binomial_distribution#Normal_approximation) or the [arcsine transformation](https://en.wikipedia.org/wiki/Binomial_distribution#Arcsine_method) to treat it this way.

So after some rewriting we have one observation in each group $X_i \sim {\mathcal N}(\theta_i, \sigma_i^2)$ and are trying to estimate the $\theta_i$.
The maximum likelihood estimate (and the best unbiased estimate) is simply $\hat\theta_i = X_i$, but that assumes we know nothing about the means.
Suppose we had some prior distribution for the $\theta_i \sim g(\cdot)$; then we could use Bayes rule to get a posterior estimate given the data:

$$\begin{align}
{\mathbb P}(\theta_i \vert X_i) &= \frac{{\mathbb P}(X_i \vert \theta_i) {\mathbb P}(\theta_i)}{\int {\mathbb P}(X_i \vert \theta_i){\mathbb P}(\theta_i)\,{\rm d}\theta_i}  \\
&= c e^{-(X_i - \theta_i)^2/(2\sigma_i^2)} g(\theta_i)
\end{align}$$

where c is a normalising constant independent of theta.
We can then summarise our estimate by a measure of centrality such as the mean, median or mode (maximum likelihood estimate).
The default method of count and divide is simply choosing an (improper) flat prior separately for each of K groups and using the maximum likelihood.
Could we do better by choosing a prior and pooling the information?

If we used our knowledge of the problem we could at least form a weak prior; the batting average of a cricket player is unlikely to be 0 or over 100 (although [Bradman](https://en.wikipedia.org/wiki/Don_Bradman) got close), a salary is unlikely to be below minimum wage (although it happens) or above a million dollars (with a few exceptions).
We could use this information to form a very weak prior that wouldn't change the estimates much, but if we have enough groups it could make a substantial difference to the estimates (this is the realm of multilevel models).
Another approach is to directly *estimate the prior from the data*, an Empirical Bayes approach; we wouldn't want to overfit to our dataset but with lots of groups and a low dimensional model it can work strikingly well.
We could look at a histogram of our maximum likelihood estimates of $\theta_i$ and guess or model the prior function g.

In particular let us assume that our means are themselves normally distributed with some unknown mean and standard deviation, $\theta_i \sim {\mathbb N}(M, A)$.
This assumption may or may not be reasonable for a given dataset; a rough check would be to plot the group means and see what the distribution looks like and choose a parametric model for it.
How much does the prior change our estimates and how to we estimate the parameters from the data?

# Posterior Estimates

To get a posterior estimate for the group mean $\theta_i$ we apply Bayes rule

$$\begin{align}
{\mathbb P}(\theta_i \vert X_i) &= \frac{{\mathbb P} (X_i \vert \theta_i) {\mathbb P} (\theta_i)}{\int {{\mathbb P} (X_i \vert \theta_i) {\mathbb P} (\theta_i)} \,d\theta_i} \\
&= c_1 e^{-(X_i - \theta_i)^2/2 \sigma_i^2} e^{-(\theta_i - M)^2/2A} \\
&= c_2 e^{-(\theta_i - B_i (X/\sigma_i^2 + M/A))^2/2B_i}
\end{align}
$$

where $c_1$ and $c_2$ are (different) normalising constants, and B is half the harmonic mean of the variances, that is $B_i = \frac{1}{1/\sigma_i^2 + 1/A}$ (the last line comes from expanding the exponent and completing the square for $\theta_i$).
So we get $\theta_i \vert X_i \sim {\mathcal N}\left(\frac{X_i/\sigma_i^2 + M/A}{1/\sigma_i^2 + 1/A}, \frac{1}{1/\sigma_i^2 + 1/A}\right)$.
Note the most likely value of $\theta_i$ is a weighted average between $X_i$ and M, weighted by the inverse variances (sometimes called precision).
So in particular for $\sigma_i^2 \ll A$ then the estimate of $\theta_i$ is very close to $X_i$, and for $A \ll \sigma_i^2$ it's very close to M.
Also as either A or $\sigma_i^2$ gets small the uncertainty in $\theta_i$ gets small.

Another convenient way to write the most likely estimate is $\frac{X_i/\sigma_i^2 + M/A}{1/\sigma_i^2 + 1/A} = X_i + (M-X_i)\frac{\sigma_i^2}{A + \sigma_i^2}$, leaving us to estimate M and $\frac{1}{A + \sigma_i^2}$.

# Estimating the parameters

We don't observe the $\theta_i$ directly, but through only their influence on the $X_i$.
So to estimate M and A it makes sense to look at the marginal distribution f of the $X_i$ over $\theta$

$$\begin{align}
{\mathbb P}(X_i \vert M, A) &= \int {\mathbb P} (X_i, \theta_i \vert M, A) \, {\rm d}\theta_i \\
&= \int {\mathbb P}(X_i \vert \theta_i) {\mathbb P}(\theta_i \vert M, A) \, {\rm d}\theta_i\\
&= \int_{-\infty}^{\infty} \frac{1}{\sqrt{2\pi \sigma_i^2}}e^{-(X_i-\theta_i)^2/2\sigma_i^2}\frac{1}{\sqrt {2 \pi A}}e^{-(\theta_i - M)^2/2A} \\
&= \frac{1}{\sqrt{2 \pi \sigma_i^2} \sqrt{2 \pi A}} e^{-(X-M)^2/(2(A+\sigma_i^2))} \int_{-\infty}^{\infty} e^{-(\theta_i - B_i (X/S + M/A))^2/2B_i} \\
&= \frac{1}{\sqrt{2 \pi (A + \sigma_i^2)}} e^{-(X-M)^2/(2(A+\sigma_i^2))}
\end{align}$$

where the fourth line comes from completing the squares of the exponents, and 
So we have marginally $X_i \sim {\mathcal N}(M, A + \sigma_i^2)$.
This is a good way to check the assumptions; the points $X_i$ (in practice the group means) should be approximately normally distributed.

In a Bayesian approach the next step would be to set a prior on M and A and then come up with posterior estimates based on the data (for example with Markov Chain Monte Carlo methods).
The difference with an Empirical Bayes approach is we estimate M and A directly with maximum likelihood; when there's a lot of signal in the data so that the resulting posterior is sharply peaked they will give approximately the same result as a fully Bayesian approach with weak priors, but the Empirical Bayes approach is a lot more computationally efficient.

The most likely values $\hat A$ and $\hat M$, are where the probability is maximised, or equivalently where the negative log likelihood is minimised.
The negative log likelihood is:

$$-l(M, A) = \frac{1}{2}\sum_{i=1}^{K} \ln(2 \pi) + \ln(A + \sigma_i^2) + \frac{(X_i - M)^2}{A + \sigma_i^2}$$

The optima occur where the partial derivatives are zero; the derivatives are:

$$\begin{align}
-\frac{\partial l}{\partial M}(M,A) &= \sum_{i=1}^{K} \frac{(X_i - M)}{A + \sigma_i^2} \\
-\frac{\partial l}{\partial A}(M,A) &= \frac{1}{2}\sum_{i=1}^{K} \frac{1}{A + \sigma_i^2} - \frac{(X_i - M)^2}{(A + \sigma_i^2)^2}
\end{align}$$

This yields coupled equations for $\hat M$ and $\hat A$


$$\begin{align}
\hat M &= \left.\sum_{i=1}^{K} \frac{X_i}{\hat A + \sigma_i^2} \middle/ \sum_{i=1}^{K} \frac{1}{\hat A + \sigma_i^2} \right. \\
0 &= \sum_{i=1}^{K} \frac{1}{\hat A + \sigma_i^2} -  \frac{(X_i - \hat M)^2}{(\hat A + \sigma_i^2)^2}
\end{align}$$

The midpoint is the weighted average of the data, weighted by the inverse *total* variance.
Substituting this into the second equation we get an equation in terms of $\hat A$ alone, but not one that I know how to solve analytically.

# Putting it all together

To recap, we started assuming that we've got K points $X_i \sim {\mathcal N}(\theta_i, \sigma_i^2)$, and we know the standard deviations $\sigma_i$ and want to estimate the means $\theta_i$.
This could arise in calculating the means of groups; when we've got enough points by the Central Limit Theorem the group sample means will be approximately normally distributed and their variances inversely proportional to the number of points (and for smaller groups for normal variables and binomial proportions).
If we further assume that the mean parameters are normally distributed $\theta_i \sim {\mathcal N}(M, A)$ then the best estimate for them is 

$$\begin{align}
\hat\theta_i &= \frac{X_i/\sigma_i^2 + M/A}{1/\sigma_i^2 + 1/A} \\
& = \omega_i M + (1-\omega_i) X_i
\end{align}$$

where $\omega_i = \frac{1/A}{1/\sigma_i^2 + 1/A} = \frac{\sigma_i^2}{\sigma_i^2 + A}$ and the best estimates for M and A are given by the equations

$$\begin{align}
\hat M &= \left.\sum_{i=1}^{K} \frac{X_i}{\hat A + \sigma_i^2} \middle/ \sum_{i=1}^{K} \frac{1}{\hat A + \sigma_i^2} \right. \\
0 &= \sum_{i=1}^{K} \frac{1}{\hat A + \sigma_i^2} -  \frac{(X_i - \hat M)^2}{(\hat A + \sigma_i^2)^2}
\end{align}$$

So the $\theta_i$ lie somewhere between the data point for the group and M, the (weighted) midpoint for all the data.
Let's analyse these in some special cases.

## Equal within-group variance

Suppose all the points $X_i$ have the same standard deviation, $\sigma_i = \sigma$, and so we have $X_i \sim {\mathcal N}(\theta_i, \sigma)$.

$$\begin{align}
\hat M &= \frac{1}{K} \sum_{i=1}^{K} X_i =: \bar X \\
\hat A + \sigma^2 &= \frac{1}{K} \sum_{i=1}^{K} (X_i - \bar X)^2 =: \frac{S}{K} \\
\hat \omega &= \frac{K\sigma^2}{S} \\
\hat \theta_i &= \omega \bar X + (1 - \omega) X_i
\end{align}$$

These estimators are biased; since $X_i \sim {\mathcal N}(M, A + \sigma^2)$ then $S := \sum_{i=1}^{K} (X_i - \bar X)^2 \sim (A + \sigma^2) \chi^2_{K - 1}$.
Given $Z \sim \chi^2_\nu$ then

$$\begin{align}
{\mathbb E}(1/Z) &= \int_0^{\infty} \frac{1}{x} \frac{ x^{\nu/2-1} e^{-x/2}}{2^{\nu/2}\Gamma(\nu/2)} \,{\rm d}x\\
&= \frac{2^{\nu/2-1}\Gamma(\nu/2-1)}{2^{\nu/2}\Gamma(\nu/2)} \int_0^{\infty} \frac{ x^{(\nu-2)/2-1} e^{-x/2}}{2^{(\nu-2)/2}\Gamma((\nu-2)/2)}  \,{\rm d}x\\
&= \frac{1}{2 (\nu/2 - 1)} \cdot 1 \\
&= \frac{1}{\nu - 2}
\end{align}$$

so an *unbiased estimator* of the $\theta_i$ is

$$\hat \theta_i = \frac{(K - 3)\sigma^2}{S} \bar X + \left(1 - \frac{(K - 3)\sigma^2}{S} \right) X_i$$

this is precisely a James-Stein estimator (when M is known it replaces $\bar X$ and removes one degree of freedom from S, meaning we get for $\sigma=1$, $\theta_i = \frac{K-2}{S} M + \left(1 - \frac{K-2}{S}\right) X_i$, which is the original estimator proposed by James and Stein which has lower risk than $\hat\theta_i^{\rm MLE} = X_i$ under the sole assumption $X_i \sim {\mathcal N}(\theta_i, 1)$).

The crucial factor here is the ratio between the variance within groups to that between groups $\frac{\sigma^2}{S/(K-3)}$.
When this is small the estimate is close to the actual points, and when this is large the estimate is close to the overall average $\bar X$.
As we more precisely estimate the group mean the more we gain from biasing towards it, and the lower the within-group variation the less there is to benefist from biasing the estimate by shrinking.

## Lower within-group variance than between-group

Suppose $\sigma_i^2 \ll A$ for all i.
Then $\hat \theta_i \approx X_i (1 - \sigma_i^2/A) + M \sigma_i^2 / A$, so the estimates are closer to the data points and the Maximum Likelihood Estimate.
In this case the benefit of this approach will be marginal.
Also $\hat M \approx \frac{1}{K} \sum_{i=1}^{K} X_i$ and $\hat A \approx \frac{1}{K}\sum_{i=1}^{K} (X_i - \hat M)^2$ as in the equal variance case.


## Lower between-group variance than within-group

Suppose $A \ll \sigma_i^2$ for all $\sigma_i$.
Then $\hat M \approx \left.\sum_{i=1}^{K} \frac{X_i}{\sigma_i^2} \middle/ \sum_{i=1}^{K} \frac{1}{\sigma_i^2} \right.$.
In particular if the $X_i$ are formed as group averages with equal variance, so $\sigma_i^2 = \sigma_0^2/N_i$, then $\hat M \approx \frac{\sum_{i=1}^{K} N_i X_i}{\sum_{i=1}^{K} N_i}$, that is it's the average of all the points themselves (instead of averages of the group centres).
The means are approximately $\hat \theta_i \approx M (1 - A/\sigma_i^2) + X_i \sigma_i^2/A$ is much closer to the mean of the data than the points themselves.

## Estimating the variances

We have assumed the $\sigma_i$ are known; in some cases like a Binomial distribution they are part of the model, but in practice they may need to be estimated.
When the $X_i$ are formed from groups with more than one data point they can be estimated over a prior distribution in a similar way to the mean.
More simply we can assume they are the same and estimate it from the mean of the sample standard deviations (correcting for sample size). 

To solve the equations for $\hat A$ an extremely low starting value would be 0.
An extremely high value would be when $\hat A \gg \sigma_i^2$, in which case we get estimates of $\hat A$ identical to the equal variance case.
In general then I'd expect the solution to be between these and could be solved by bisection (though I'm not completely sure of the upper bound, or that it has a unique solution).

# What's next

This gives some of the theory of estimating the means of groups, but it would be good to exercise it against some practical examples.
The benefits are largest when the between-group variance is lower than within-group variance, so in particular when there are lots of groups and a few noisy estimates in each group.
The estimators we derived are biased, but it's not a big issue when there are a large number of groups (say when $K \gg 3$).

The amount of shrinking changes with the standard deviation which can change the ranking of the results.
This provides a methodological tool that covers a substantial problem in analysis; the most extreme data points tend to come from the data with most variation.
This could potentially be used to address these problems (and even construct probabilistic orderings) in a principled way. 

It could also be interesting whether we could construct Empirical Bayes models for other distributions, in particular binomial distributions could be interesting (when some observations only have a few data points and the normal approximation breaks down).

Finally we could take this further and build, and estimate, multilevel models directly.
However these are harder to analyse in general and a lot more computationally intensive.