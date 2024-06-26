---
categories:
- maths
- data
date: '2021-10-10T20:48:10+11:00'
image: /images/stein.png
title: A Reading Guide to Stein's Paradox
---

[Stein's Paradox](https://en.wikipedia.org/wiki/Stein%27s_example) states that when trying to estimate the 3 or more means of normally distributed data together, it's *always* better (on average) to shrink the estimates.
Specifically if you've got p independent normally distributed variables $X_i \sim N(\theta_i, 1) ;\, i=1,\ldots,p$ the best estimates for minimising the mean squared error of *all* the estimates isn't the values themselves $X$, and the James-Stein estimator is better (has strictly lower risk).

$$\hat\theta^{JS}(X) = \left(1 - \frac{p-2}{\lVert X\rVert^2}\right)X$$

A lot of the details here can be weakened substantially.

This article will give a guide on how to understand this phenomenon a bit better.

# What is Stein's Paradox?

The best introductory resource is this [Statslab Cambridge](http://www.statslab.cam.ac.uk/~rjs57/SteinParadox.pdf) article by Richard Samsworth, which gives a clear explanation and a very simple proof.
If the notation is a bit hard to follow I recommend the book [All of Statistics](http://www.stat.cmu.edu/~larry/all-of-statistics/index.html) which covers Decision Theory in Chapter 10 (and touches on the James-Stein Estimator).

One thing to note from this article is the improvement in risk over the Maximum Likelihood Estimator $X$ is $(p-2){\mathbb E}\left(\frac{1}{\lVert X \rVert^2}\right)$.
So the closer the points are to the origin the more the improvement (although there is always *some* improvement).
And since the choice of origin is arbitrary (through a change in coordinates) having a good guess of where to shrink the estimates to will give a much better result like in the baseball example.

# Why is it important?

Bradley Efron has done a lot of writing connecting it to *empirical Bayes* methods, showing it as a striking example of how on large datasets blending Bayesian methods with frequentist estimates can lead to striking solutions.
The 1977 Scientific American Article [Stein's Paradox in Statistics](https://statweb.stanford.edu/~ckirby/brad/other/Article1977.pdf) by Efron and Morris gives a good flavour of what it means an why it's important.
They state a slightly different estimate (where $X_i \sim N(\theta_i, \sigma)$):

$$\hat\theta^{JS} = \bar X + \left(1 - \frac{(p-3) \sigma^2}{\lVert X - \bar X \rVert^2} \right) (X-\bar X)$$

here instead of picking the origin they're estimating it from the data as the *grand mean* $\bar X = \frac{\sum_{i=1}^{p} X_i}{p}$, which will give a better than random risk but at the cost of 1 degree of freedom (the p-3 in the numerator instead of p-2).
The other thing to note is the larger the standard deviation the more we shrink the estimate (which makes sense since we are less certain about it).
I suspect if they had different standard deviations you would shrink more in directions with larger standard deviation.

To understand this connection Chapter 1 of Efron's [Large-Scale Inference](https://statweb.stanford.edu/~ckirby/brad/LSI/monograph_CUP.pdf) gives a very good introduction.
It walks through how starting with the model $\theta \sim N(0, A)$ and $X \vert \theta \sim N(\theta, 1)$ the James-Stein estimator can be recovered, and then how it can be extended to estimate the mean or standard deviation.
It also explains *limited translation estimators* where we shrink less, which gives a higher risk but less biased estimator.
A similar (but briefer) explanation is in Chapter 7 of [Computer Age of Statistical Inference](https://web.stanford.edu/~hastie/CASI/) which also covers the connection with ridge regression.
A great more general tutorial is Casella's [An Introduction to Empirical Bayes Data Analysis](https://www.biostat.jhsph.edu/~fdominic/teaching/bio656/labs/labs09/Casella.EmpBayes.pdf).

# But why does it work?

The connection to Empirical Bayesian methods gives useful applications, but it doesn't indicate why it works.
The best heuristic explanation I've seen is in [A Geometrical Explanation of Stein Shrinkage](https://projecteuclid.org/download/pdfview_1/euclid.ss/1331729980) by Brown and Zhao (2012), which shows how shrinking allows reducing the variance in the other dimensions ([Joe Antognini](https://joe-antognini.github.io/machine-learning/steins-paradox) has a good web article summarising this).
Naftali Harris has a [great visualisation](https://www.naftaliharris.com/blog/steinviz/) of the shrinkage, and the argument reminds me of a how volume increases quickly with dimension.

# But really, why does it work?

The paper [Admissable Estimators, Recurrent Diffusions and Insoluble Boundary Value Problems](http://stat.wharton.upenn.edu/~lbrown/Papers/1971b%20Admissible%20estimators,%20recurrent%20diffusions,%20and%20insoluble%20boundary%20value%20problems.pdf) by Brown (1971) connects the admissibility of estimators to recurrence of Brownian motion.
I haven't dug deep enough into the paper to understand it but it sounds mathematically deep and gives an idea as to why 3 is the critical dimension.

# How did it all start?

The original paper showing inadmissibility is [Inadmissibility of the usual estimator for the mean of a multivariate normal distribution](https://projecteuclid.org/download/pdf_1/euclid.bsmsp/1200501656) by Stein (1956).
This was followed with the explicit estimator in [Estimation with Quadratic Loss](https://projecteuclid.org/ebook/Download?urlId=bsmsp%2F1200512173&isFullBook=False) by James and Stein (1961).
The papers are certainly readable, but I found the earlier papers got to the point more succinctly.
There was a lot of follow up papers at the time on improving the estimate, connecting it with Bayesian estimators and the like but they don't strike me as deeply.

# Where next?

Stein's result really is still surprising to me; the best estimator in high dimensions are biased estimators *no matter where you bias it*, but it seems to have to do with removing some of the variance inherent in estimating multiple points in higher dimensional spaces.
However for practical applications the biggest difference is when they are (unsurprisingly) biased towards their true values, which brings us back to things like hierarchical models and Empirical Bayesian methods.

However this seems like a far reaching result that should change the practice of analysts; in a sense it's another kind of regression to the (grand) mean.
Whenever I'm calculating averages for lots of groups to maximise predictive accuracy I should shrink the estimates, and the shrinkage should increase with the variance.
I think this gives a better solution to Evan Miller's [How not to sort by average rating](https://www.evanmiller.org/how-not-to-sort-by-average-rating.html); instead of using (arbitrary) confidence intervals we shrink towards the grand mean with more shrinkage the more ratings an item has.
However I wouldn't use the James-Stein estimator directly, but instead use an empirical Bayes method.
It seems strange it's not more widely advocated, especially as a step towards machine learning based methods.

I find it interesting that even though the theorem is for normally distributed variables, the most common example are binomials from baseball (after a suitable transformation).
I wonder what an empirical Bayes method would look like, and whether it would have a lower risk than the proportion of true results.
