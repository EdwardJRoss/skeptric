---
categories:
- data
date: '2020-04-10T21:27:32+10:00'
image: /images/all_of_statistics.jpg
title: All of Statistics
---

For anyone who wants to learn Statistics and has a maths or physics I highly recommend [Larry Wasserman's *All of Statistics*](https://www.stat.cmu.edu/~larry/all-of-statistics/) .
It covers a wide range of statistics with enough mathematical detail to really understand what's going on, but not so much that the machinery is overwhelming.
What I learned reading it really helped me understand statistics well enough to design bespoke statistical experiments and effectively use and implement machine learning models.

As I was working more in analytics I found that I needed to understand more about statistics.
People around me were talking about ANOVA and I didn't know what that meant (was it something to do with linear regression?)
I also had a friend recommend me [The Elements of Statistical Learning](https://web.stanford.edu/~hastie/ElemStatLearn/), but I didn't have enough statistics to get through the first chapter.

After researching online I bought [Statistics in a Nutshell](http://shop.oreilly.com/product/0636920023074.do).
This is a good practical reference, it covers many situations and discusses the techniques to use with worked examples.
But I felt I got really stuck on *why* all these different tests existed.
It seemed like you needed a reference guide to tell you what tests to perform - but what if you were doing something that wasn't in the guide.

Then I came across [Larry Wasserman's *All of Statistics*](https://www.stat.cmu.edu/~larry/all-of-statistics/).
I spent about 18 months working through it (on trains, in coffee shops before work and at home in the evenings).
I have 4 notebooks working through the exercises and [the code exercises](https://github.com/EdwardJRoss/all_of_statistics_exercises).
It enabled me to understand how to use different statistical tests, gave me the terminology to read The Elements of Statistical Learning and broadened my statistical horizons.

Here are some highlights of things I have used since reading the book:

* Algebraic tricks for expectations and variances like $V(X + Y) = V(X) + V(Y) + 2 \rm{Cov}(X, Y)$ (which are helpful when reading books/papers)
* Bootstrap confidence intervals are very powerful for calculating statistics on large datasets where the central limit theorem doesn't apply
* Maximum Likelihood Estimators as a tool for fitting parametric models
* Hypothesis tests essentially *cut* a confidence interval; the various statistical tests are (estimates of) confidence intervals for different distributions
* When doing multiple tests you need to adjust your error rate using something like a [Bonferroni Correction](https://en.wikipedia.org/wiki/Bonferroni_correction)
* How ggplot's [density plots](https://ggplot2.tidyverse.org/reference/geom_density.html) use Kernel Density Estimation and to always set the [bandwidth](https://www.rdocumentation.org/packages/stats/versions/3.6.2/topics/bandwidth) to `ucv` or `SJ`
* Implementing [tree algorithms](https://github.com/EdwardJRoss/all_of_statistics_exercises/blob/master/chapter_22.R#L176) and understanding the order of factors of a categorical variable matters
* Learning about bagging and boosting

There's still a lot of the book I haven't used much, especially Statistical Decision Theory and Causal Inference, but I got a lot out of working through the book.
More importantly it gave me the tools and language to be able to practice statistics and machine learning.