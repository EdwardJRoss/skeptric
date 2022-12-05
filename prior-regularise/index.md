---
categories:
- data
date: '2021-08-07T08:00:00+10:00'
image: /images/regularization.svg
title: Priors as Regularisation
---

In Bayesian statistics you have to choose a prior distribution for the parameters to combine with the data to get a posterior distribution.
Choosing a tight prior, assuming that the parameters should live in a particular space, reduces the impact of the data on the posterior estimates.
This is just like [regularisation](https://en.wikipedia.org/wiki/Regularization_(mathematics)) in machine learning where adding a penalty to the loss function prevents over-fitting.
This is more than just an analogy, and this article will explore a couple of cases with constant regression and classification.

A typical machine learning approach to regression is to minimise the root mean squared error.
A probabilistic perspective for this is to consider the regression $$ y = f_\theta(X) + \epsilon $$, where y is the outcome, X are the predictors, $$ f_\theta $$ is a function parameterised by $$\theta$$, and $\epsilon$ is the error term.
If we assume that $$\epsilon \in N(0, \sigma^2)$$ is normally distributed, this is equivalent to saying that $$ y \in N(f_\theta^2(X), \sigma^2) $$.
We then need to pick the most likely parameters $$\theta$$ given the data.

The Bayesian perspective on this is if we have a prior on the parameters $$ p(\theta) $$, and data $$X_i, y_i$$ then the posterior estimate is $$ p(\theta \vert X_i, y_i) = \frac{p(X_i, y_i \vert \theta) p(\theta)}{p(\{X_i,y_i\}_i)} $$.
In Bayesian statistics we estimate the whole distribution, but we can focus on the maximum likelihood estimator, the value of $$\theta$$ that maximises the posterior probability.
Since the logarithm is a monotonic function, the maximum likelihood occurs as the same point as the maximum log likelihood.
Taking the logarithm and plugging in the normal distribution for $$ p(X,y \vert \theta) $$ gives $$ l(\theta, \sigma) = -\frac{1}{2\sigma^2} \sum_{i=1}^{N} (f_{\theta}(X_i) - y_i)^2 + \log(p(\theta)) - N \log(\sigma) + c $$ for some constant c.
In the case of a flat prior, $$ p(\theta) = 1 $$ then the maximum likelihood estimator is equivalent to minimising the (root) mean squared error.
However in general the prior acts as a regularisation; for example if we take a prior that the parameters are normally distributed it reduces to [Tikhonov Regularisation](https://en.wikipedia.org/wiki/Tikhonov_regularization).
However we could pick other prior [distributions](/distribution-between-mean-median) to recover an Lᵖ regularisation, and in particular a [Laplace distribution](https://en.wikipedia.org/wiki/Laplace_distribution) recovers the [LASSO](https://en.wikipedia.org/wiki/Lasso_(statistics)).

There's more here, in Bayesian statistics people tend to use a Horseshoe Prior instead of a Laplace Distribution, and Michael Betancourt has an article on my reading list on [Bayes Sparse Regression](https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html) that goes through the trade-offs with different regularising priors.

# Binary classification

Similar ideas can be applied in Binary Classification, here the metric is typically [Binary Cross Entropy](https://en.wikipedia.org/wiki/Cross_entropy).
From a probabilistic perspective we can assume the data comes from a Binomial distribution $$ y \in B(f_\theta(X)) $$.
Here $$ p(X_i, y_i \vert \theta) = f_\theta(X_i)^{y_i} (1 - f_\theta(X_i))^{1-y_i}$$ (keeping in mind that $$y_i$$ can only take the values 0 or 1).
Then, as in the normal regression case, we can find the maximum likelihood estimator by minimising the log likelihood $$ l(\theta) = \sum_{y_i = 1} \log(f_\theta (X_i)) + \sum_{y_i=0} \log(1 - f_\theta(X_i) + \log(p(\theta)) + c$$.
With a flat prior this maximising the log likelihood is equivalent to minimising the Binary Cross Entropy.

Consider in particular the [constant model](/constant-model) $$ f_\theta(X_i) = \theta $$, where this reduces to $$l(\theta) = s \log(\theta) + (N-s) \log(1-\theta) + \log(p(\theta)) $$, where s is the number of successes and N is the total number of trials.
A bit of calculus and algebra shows that with a flat prior this is maximised when $$ \hat{\theta} = \frac{s}{N} $$.

One problem with this is the [variance of the binomial](/bernoulli-binomial) is $$ \sqrt{\frac{\theta(1-\theta)}{N}} $$, and so if we have 0 or N successes the maximum likelihood estimate for the variance is 0, which in most cases isn't right - we're not going to be exactly zero.
A method for handling this, which I learned in the book Regression and Other Stories, is to set a prior of $${\rm Beta}(3,3)$$ which is equivalent to adding 4 extra trials with 2 successes.
Then the maximum likely estimate for the parameter is $$ \hat{\theta} = \frac{s+2}{N+4} $$ and the variance will always be non-zero.

In the log likelihood this adds a penalty of $$ \log(\theta^2 (1-\theta)^2) + c'$$, for some constant $$c'$$.
Rewriting $$ \psi = \theta - \frac{1}{2}$$ and rearranging gives the penalty, up to a constant, as $$ 2 \log(\frac{1}{4} - \psi^2)$$.
For small $$\psi$$ we can do a Taylor expansion to get $$-8 \psi^2 = -8 (\theta - \frac{1}{2})^2 $$.
So this transformation is similar to a $$l^2$$ penalty (I suspect this is for the same reason a Binomial converges to a Gaussian for large samples and moderate probabilities).

What's interesting here is the Beta prior gives a more reasonable and understandable regularisation than $$l^2$$ regularisation, especially for probabilities close to 0 or 1.
I would never have thought of a log Beta penalty, but thinking of it as a prior it makes really good sense.
On the other hand being able to switch to a maximum likelihood, and thinking of the prior as a penalty, makes things much quicker to calculate than trying to estimate the whole posterior.
There's a Wikipedia article on [Bayesian interpretation of Kernel Regularisation](https://en.wikipedia.org/wiki/Bayesian_interpretation_of_kernel_regularization)
It's useful being able to switch between the two viewpoints.