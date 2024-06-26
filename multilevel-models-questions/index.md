---
categories:
- statistics
date: '2021-09-17T12:34:20+10:00'
image: /images/heirarchical.webp
title: Learning about Multilevel Models
---

The concept of a [*multilevel model*](https://en.wikipedia.org/wiki/Multilevel_model), also called a mixed effects model or a hierarchical model, is reasonably new to me.
It's not the kind of thing typically taught in physics (where there are very explicit models) or in machine learning, but is quite common in social science.
I first came across it through [Lauren Kennedy on the Learning Bayesian Statistics Podcast](https://www.learnbayesstats.com/episode/34-multilevel-regression-post-stratification-missing-data-lauren-kennedy), through talking with a trained neuroscientist and a trained statistician who were talking about fixed and variable effects as I went cross-eyed, and through the excellent [Regression and Other Stories](https://avehtari.github.io/ROS-Examples/) textbook which makes many allusions to it (to be expounded on in their upcoming sequel [Applied Regression and Multilevel Models](www.stat.columbia.edu/~gelman/armm/)).
The exposition in Chapter 9 of [Kruschke's Doing Bayesian Data Analysis](http://doingbayesiandataanalysis.blogspot.com/) gives a good introduction in the binomial case, but I still don't really understand multilevel models and this article will list some questions I'm trying to understand.

# Overview of multilevel models

Suppose there are a number of different categories, some of which are quite small, and we want to estimate the probability that a sample from a given category is positive.
The simplest [constant model](/constant-models) is to ignore the categories and just estimate the population proportion (that is the fraction of *all* items that is positive), which has high bias but low variance.
Another approach is to calculate the proportion for each category independently (or equivalently, fit a linear or logistic regression on the one-hot encoded categorical variable) which has high variance but low bias.
Since the standard error in the estimated proportion [is](/bernoulli-binomial) $\sqrt{\frac{p(1-p)}{N}}$ where p is the true proportion and N is the number of items in the category, for small catagories this uncertainty is going to be high (another way of looking at it; if you've only got 2 data points the only possible proportions are 0%, 50% or 100%).
Ideally we'd like to interpolate between the two; if the overall average is say 14% and the category only has a few observations we'd like to estimate something close to 14%.
On the other hand if we have lots of data we'd like to get something very close to the category's own proportion.
How heavily we rely on the population estimate would depend on how much variance there is between categories.

A statistical way of framing this is to say each category, C, is Bernoulli distributed with it's own probability $p_C$, so the individual items in category C are $X_C \sim {\rm Binom}(p_C)$.
However these aren't independent, when we see a new category it's likely close to the typical category probability, and so they come from some common distribution.
One possibly choice of distribution is a Beta Distribution; then $p_C \sim {\rm Beta}(A, B)$ for some A and B (although there are other choices of distribution).
This has a nice Bayesian interpretation; for each category C we estimate the Binomial distribution with a Beta(A, B) prior.
In any case we need to estimate $p_C$ as well as A, B, which leads to [shrinking](doingbayesiandataanalysis.blogspot.com/2012/11/shrinkage-in-multi-level-hierarchical.html) the probabilities towards the typical probability across categories.

This can be extended to multiple hierarchical structures and levels of hierarchy.
A very common example in the literature is interventions in students.
If an educational treatment is being exposed to students in classrooms, students in the same classroom are more similar than those in different classrooms (since they have the same teacher and would be exposed to the treatment in the same way).
For a linear regression it may not be reasonable to treat students in the same classroom as independent and instead we need to somehow account for the correlation between students in the same classrooms, and a multilevel model allows us to do this.

In general there may be multiple levels of heirarchy (for example children in classrooms in schools in districts), or multiple simultaneous heirarchies (for example music students are in a classroom but they may also be grouped by instrument), and the underlying models do not need to be linear.

# Questions about multilevel models

I'm interested in multilevel models in a machine learning context.
I have been using some high cardinality categorical data with millions of observations and am interested in whether multilevel models could improve the estimates (or whether it's just better to use [categorical embeddings](/categorical-embeddings) in these cases).

## How do we estimate the models?

These can be estimated using Bayesian methods, but for large datasets and complex heirarchies MCMC based approaches can be computationally intractable.
I'm not sure whether they can be structured in a way to make them faster, or perhaps using good priors or approximate Bayesian approaches could help.
Is there are way to estimate heirarchical models on large datasets using Bayesian methods in reasonable time?

Another approach is Maximum Likelihood Estimation where we find the most likely parameters (and Bayesian priors can be incorporated [as regularisation](/prior-regularise)).
This boils down to function optimisation which can be done efficiently with blackbox methods.
Given we have a large amount of data do we lose anything with these methods over Bayesian approaches (I would expect the modes to be quite sharp, depending on the complexity of the model)?
I've also heard of Restricted Maximum Likelihood Estimation - what is that and how does it compare?

## Can we include information about the correlation?

If we have lots of information about the categories is there a way to encode this in the models (perhaps as priors or as extra predictors)?

## What software is there for these models?

For a likelihood approach [lme4](https://github.com/lme4/lme4) seems to be the staple; how do we use it?
Is there an equivalent in Python; statsmodels [MixedLinearModel](https://www.statsmodels.org/stable/mixed_linear.html) looks a lot less flexible?

For Bayesian methods there's Stan, PyMC3, JAGS among other software (along with wrappers like [`brms`](https://github.com/paul-buerkner/brms) that provides a formula syntax close to lme4 in Stan).
How does the performance compare between them, and how can they be implemented in an efficient way?

## Using with non-linear models?

I wonder whether heirarchical shrinkage would be useful in tree based models (and perhaps make their estimates more stable along with less variable selection like [ctrees](https://cran.r-project.org/web/packages/partykit/vignettes/ctree.pdf)).
In general heirarhcical shrinkage can be added to Bayesian models by having higher level distributions of the parameters, does it make sense to do this in practice?

## How does shrinkage vary with distribution?

It would make sense that the higher the interclass deviation and the smaller the intraclass deviation the more shrinkage there should be to the group parameters.
Does this happen? Can we quantify this?

# Next steps

I'll start reading some basic papers on the topic to get an understanding of these questions and try to set up some analysis to help answer them.
