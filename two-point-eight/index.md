---
categories:
- data
- statistics
date: '2021-04-08T08:00:00+10:00'
image: /images/power_ci.png
title: 'Statistical Testing: 2.8 Standard Deviations'
---

What sample size do you need to capture an effect with 95% confidence 80% of the time?
For a normal/binomial distribution the answer is roughly $$ \left(2.8 \frac{\sigma}{\epsilon}\right)^2 $$, where $$ \sigma $$ is the standard deviation of the data and $$ \epsilon $$ is the size of the effect.

The ratio $$ \frac{\sigma}{\epsilon} $$ says the smaller the effect size is relative to the variability in the data the larger the sample size you will need.
The quadratic dependence means that if you can double the effect size you only need 1/4 of the sample; wherever possible maximise your effect.

But why the factor 2.8?
For a 95% confidence the measured value must be 1.96 standard deviations from the mean.
But there's variability in the measure itself - if the true effect size is $$ \epsilon $$ the measurement will be drawn from a normal distribution centred at $$ \epsilon $$ with standard deviation $$ \sigma $$.
So using 1.96 as the sample coefficient would only measure the effect half the time.
We need to increase our sample size so that two standard deviations occurs at the 20th percentile of the distribution for $$\epsilon$$; namely 1.96 + 0.84 = 2.8 standard deviations.
More generally the coefficient for $$\alpha$$ confidence at $$\beta$$ power is $$ z_{1 - \alpha / 2} z_{1 - \beta} $$, or in R `qnorm(1-(1-alpha)/2) + qnorm(1-(1-beta))`.

As a side note you shouldn't skimp on power.
If your study is under-powered you're much more likely to overestimate the effect size or even get the sign wrong when you do find a significant result.
Andrew Gelman and John Carlin's paper [Beyond Power Calculations](http://www.stat.columbia.edu/~gelman/research/published/retropower_final.pdf) go into the detail; essentially because a significant result has to be 2 standard deviations from the mean, if you're unlucky enough to see a spurious "statistically significant" result, the apparent effect size will always be far too large.

And do you really want to minimise type 1 errors anyway?
When you're making decisions you typically want to make the best decisions subject to constraints, and often less precise estimates of effects is good enough (it's easy for the effect size to be below practical significance).
An example of this approach is [Test and Roll](/test-and-roll) that first "learns" the best alternative with an A/B test and then "earns" by rolling it out to the rest of the population; but rather than picking sample sizes for 95% confidence, they pick sample sizes to maximise expected returns.