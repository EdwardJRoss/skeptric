---
categories:
- data
- statistics
date: '2021-04-10T08:00:00+10:00'
image: /images/binomial_sample_size.png
title: How big a sample to measure conversion?
---

A common question with conversions and other rates, is how big a sample do you need to measure the conversion accurately?
To get an estimate with standard error $\sigma$ you need at most $\frac{1}{4 \sigma^2}$ samples.
In general if the true conversion rate is p it is $\frac{p(1-p)}{\sigma^2}$.

So let's say we want to measure the conversion rate within about 5%.
To be conservative we'd want the standard error to be a bit less than that, say 3%
Then we would need at least  $\frac{1}{4 (0.03)^2}  \approx 278$ samples.
Note that to double the precision we need to quadruple the sample.

If you want to run a null hypothesis test with 95% CI and 80% power then you need to multiply by the square of
[2.8](/two-point-eight).
That is $\frac{(2.8)^2}{4 \sigma^2}$, where $\sigma$ is the detection size.
For an A/B test we require double this (since the variances add); so to see an uplift of 5% would require $\frac{2 (2.8)^2}{4 (0.05)^2} = 1570$ in *each* group (actually a little more due to the [continuity correction](https://en.wikipedia.org/wiki/Continuity_correction).
If this seems too big maybe you don't actually want a significant test; look into something like [test and roll](/test-and-roll).

For the details on why this is read [from Bernoulli to the Binomial](/bernoulli-binomial)