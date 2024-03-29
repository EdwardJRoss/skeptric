---
categories:
- data
date: '2021-03-01T18:39:28+11:00'
image: /images/binomial_likelihood.svg
title: Metrics for Binary Classification
---

When evaluating binary classifier (e.g. will this user convert?) the most obvious metric is *accuracy*; what's the probability a random prediction is correct.
One issue with this metric is if 90% of the cases are one class a high accuracy isn't really impressive; you need to contrast it with a [constant model](/constant-models) predicting the most frequent class.
More subtly it's not a very *sensitive* measure, by measuring cross-entropy of predicted probabilities you get a much better idea of how well your model is working.

Consider the case of predicting people who will purchase a product.
Then the estimated revenue is the sum of the price of each product multiplied by the probability they will buy it.
Here you get a much better estimate by using the probability; you could get a distribution of that estimate by simulating the picks at each probability (for example high price, low probability purchases could add a lot of uncertainty to the predictions).
This is much more useful than the binary prediction of whether they will buy the product, which will give an extreme estimate.

One way to evaluate the prediction of the probabilities of the outcomes is via *likelihood*.
For a [bernoulli distribution](/bernoulli-binomial) with constant, but unknown, probability p, the probability of getting a positive is p and a negative is (1 - p).
Suppose we've taken N draws and got S positive cases, the likelihood of this result for a given p is $L(p) = p^S (1 - p)^{N-S}$.
Then we're trying to work out what the probability p is given the data.
One way to answer this is to pick the value of p which is most likely, which happens to be S/N (i.e. the average number of positive cases, as you would expect).

Instead of the likelihood, the log-likelihood is often used.
It has a maximum at the same point as the likelihood, but because probabilities are multiplicative, log-probabilities are additive and so it's often simpler and more numerically stable.
For the Bernoulli distribution example above the log likelihood is $l(p) = S \log(p) + (N-S) \log(1-p)$.
This log likelihood can also be used as a *measure* of how good our model is; since it's minimum is the most likely model.

This can be generalised to the case where we have a different probability for each data point.
Then the log likelihood generalises to $l = \sum_{i \in +} \log(p_i) + \sum{i \in -} \log(1 - p_i)$ where `+` is the set of positive results and `-` is the set of negative results.
Another way of writing this is $l = \sum_i q_i \log(p_i) + (1 - q_i) \log(1 - p_i)$, where q is 1 for a positive case and 0 for a negative case; in this form it is clear it sums the log of the probability of the actual outcome (and can be generalised to more than 2 outcomes).
The negative of this is called the *cross entropy*, and is a measure of how good our predictions are (since cross entropy is smaller the more likely the model is).
In code this is:

```
def binary_cross_entropy_item(probability, actual):
    if actual == 1:
      return -log(probability)
    else:
      return -log(1 - probability)
      
def binary_cross_entropy(probabilities, actuals):
    return sum([cross_entropy_item(prob, actual) for prob, actual in zip(probabilities, actuals)])
```

## Cross entropy versus mean squared error

How does cross-entropy compare to using the (root) mean squared error on the probabilities?
For a given data set from a normal distribution with a known standard deviation the log likelihood is a constant plus a term proportional to the squared mean error.
Precisely it's

$$l(\mu, \sigma) = -n \log \sigma - \frac{n S^2}{2 \sigma^2} - \frac{n(\bar{X} - \mu)}{2 \sigma^2}$$

where $\bar{X} = \frac{1}{N} \sum_{i=1}^{N} X_i$ is the sample mean and $S = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (X_i - \bar{X})^2}$ is the sample standard deviation.
So for a given $\sigma$ the log likelihood only depends on the mean square error.

As the number of trials increases the binomial distribution gets close to a normal distribution (but slower for more extreme p).
So using the mean square error isn't crazy, for large samples you'll get reasonable results, but it's not as appropriate as the cross entropy.

## Further reading

This barely scratches the surface of loss functions for classification; [Wikipedia has an article](https://en.wikipedia.org/wiki/Loss_functions_for_classification) that covers other kinds of loss such as exponential loss, tangent loss, hinge loss, and savage loss.
It would be great to understand more about how these all differ and how to choose between them in a particular application.