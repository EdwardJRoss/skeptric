---
categories:
- data
date: '2021-04-06T19:41:49+10:00'
image: /images/test_roll_website_example.webp
title: More Profitable A/B with Test and Roll
---

When running an A/B test the sample sizes can seem insane.
For example to observe a 2 percentage point uplift on a 60% conversion rate requires over 9,000 people in *each* group to get the standard 95% confidence level with 80% power.
If you've only got less than 18,000 customers you can reach, which is very common in businesss to business settings, it's impossible to conduct this test.
But if you look in terms of the outcomes, doing an A/B test on a few hundred users may actually greatly increase your outcomes.

[Test & Roll: Profit-Maximizing A/B Tests](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3274875) by Elea McDonnell Feit and Ron Berman tackles this problem giving estimates for small samples.
In the context of a one-off opportunity (like advertising, promotional offers, campaign launches, products that will only be bought once) we actually don't *care* if one version is significantly better than the other; all we care about is maximising our outcome.
Standard practice is to run an A/B test to determine which alternative is better, and then roll it out to 100%.
There's a tradeoff in how long we explore to find the optimal alternative, and how long we have left to exploit this opportunity.

They tackle this problem with Bayesian Decision Theory.
They run both A and B in parallel on a limited sample, until we get enough evidence to just pick one.
But here enough evidence means "good enough" to run with; if they're very similar then it doesn't matter too much and instead of running a high fidelity statistical test to get evidence on which is better, just pick one.
In particular they look to maximise the *expected* (average) return, which makes sense if you're running lots of these processes.
If you have a rough idea of how the data is distributed and the size of the difference between them (something you need for a classical statistical test), you can work out the optimal sample size for your A/B test, before rolling it out to 100%.

In particular they assume that the returns are normally distributed.
In my experience if there are opportunities to make multiple purchases the returns tend to be long tailed; there's a few big spenders and a lot of low spenders.
But it's a fine approximation for one-off purchases and conversions (since binomials can be approximated by normal distribution), and the framework can be extended to this case.
They define variables for a symmetric A/B test: $$Y_{A} \sim \mathcal{N}(m_A, s^2)$$, $$Y_{B} \sim \mathcal{N}(m_B, s^2)$$ where $$ m_{A}, m_{B} \sim \mathcal{N}(\mu, \sigma^2)$$.
So in the case of a conversion rate:

* $$Y_A$$ and $$Y_B$$ represent the results we get from alternative A and B respectively
* $$m_A$$ and $$m_B$$ are the true conversion rates of each alternative
* $$\mu$$ is the expected conversion rate
* s is the standard deviation of the data; it's approximately $$\sqrt{\mu(1-\mu)}$$
* $$\sigma$$ is the expected variation in the conversions between the conversion rate

The hardest of these parameters to understand is $$\sigma$$; one way to understand it is the expected difference of $$\vert m_A - m_B \vert $$ is $$ \frac{2}{\sqrt{\pi}} \sigma $$ (note in the paper the square root in the numerator is a typo!).
So if we expect the difference between A and B to be about 2 percentage points, then we should set $$ \sigma $$ as $$ 0.02 \frac{\sqrt{\pi}}{2} \approx 0.018 $$.
In general $$\sigma$$ is around 89% of the mean difference, and 95% of the median difference, so it's pretty close to the expected effect size.

Under these conditions they find the optimum sample size for each of the A and B groups are:

$$ \sqrt{\frac{N}{4} \left(\frac{s}{\sigma}\right)^2 + \left(\frac{3}{4} \left(\frac{s}{\sigma}\right)^2\right)^2} - \frac{3}{4} \left(\frac{s}{\sigma}\right)^2 $$

It's interesting to note the key variable is the ratio $$\left(\frac{s}{\sigma}\right)^2$$.
The smaller our effect size the larger the sample we need, the smaller the variability in the data the smaller sample size we need.
In our conversion example above this ratio is around 600.

In the limit of very large populations, large effect sizes and small variation $$ N \gg \left(\frac{s}{\sigma}\right)^2 $$, the sample size approaches $$\sqrt{N} \frac{s}{2 \sigma}$$ from below.
In the limit of very small populations, small effect sizes and large variation $$ N \ll \left(\frac{s}{\sigma}\right)^2 $$ it approaches $$ N/6 $$.
Finally for $$ N = \left(\frac{s}{\sigma}\right)^2 $$ the optimal sample size is $$\frac{\sqrt{13} - 3}{4} N \approx 0.15 N $$.

So for small and moderate population sizes (relative to $$  \left(\frac{s}{\sigma}\right)^2 $$ ) the optimum process is to run both A and B on about 1/6 of the population each, and then run the winner on the remaining 2/3 of the population.
As the population increases, effect size increases, or variation decreases the size of the A/B tests decreases, and in the limit scales as $$\sqrt{N}$$, and so we can run the test on a small sample.

The upshot of all this is on average we capture some percentage of the expected improvement.
Indeed the expected profit with this optimum sample sizes in each group A and B, n* is

$$ N \left(\mu + \frac{2}{\sqrt{\pi}} \sigma \left(1 - \frac{2n^*}{N} \right) \frac{1}{\sqrt{2 + \frac{4}{n^*} \left( \frac{s}{\sigma}\right)^2}}  \right) $$

As explained the term $$\frac{2}{\sqrt{\pi}} \sigma $$ is the optimum expected gain.
This is then attenuated by the exploration cost $$ \left(1 - \frac{2n^*}{N} \right) $$, which is the percentage of cases not in the test phase.
Finally it's attenuated by the expected cost of picking the wrong alternative due to chance, $$\frac{1}{\sqrt{2 + \frac{4}{n^*} \left( \frac{s}{\sigma}\right)^2}} $$.
For very low populations or small effect sizes relative to the variation in the data, the gain over random choice $$ N \mu $$ is small.
But as the effect size increases and populations increase this procedure captures more of the gains.

There's a good [blog post from co-author Ron Berman](https://www.ron-berman.com/2020/01/26/test-and-roll/) that explains this further, and [an online sample size calculator](https://testandroll.com).
In the paper they compare it to a Multi-Armed Bandit with a Thompson sampler, which does better, but is much more complex to implement and control.
I really like the framework of Bayesian Decision analysis; it seems flexible enough that a different set of assumptions or goals could be easily implemented.

My biggest concern with this procedure is choosing priors, and the effect of choosing bad ones.
If your prior deviation is too small you'll test longer and gain less; there may be worse effects if you pick the wrong distribution.
Because we're not performing a whole test we also will never really know if we've chosen the better alternative, and so it's hard to incorporate this feedback into updating priors.
It's going to be really hard to know if we're optimising correctly.

Another consideration is what to do with a situation where there are repeat purchases on a long-lived website.
The test-and-roll framework doesn't really apply, but there are still opportunity costs with testing and rolling out.
[Chris Said has a blog series](https://chris-said.io/2020/01/10/optimizing-sample-sizes-in-ab-testing-part-I/) which comes to similar conclusions but frames it in terms of the *time* opportunity cost - I look forward to reading this more.

I'm also curious what happens if we've got multiple alternatives.
There are a few scenarios where we could have many similar effect size alternatives, or quite risky alternatives, and I wonder what the best procedure is that maximises expected return.
Also for uncommon opportunities we may be more conservative and consider the whole distribution; maybe we maximise the chance of hitting some revenue target, or maximising a percentile of returns.
These should be straightforward extensions of the theory, and could be evaluated numerically through simulations.