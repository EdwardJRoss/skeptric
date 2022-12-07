---
categories:
- stan
- r
date: '2021-08-14T22:13:04+10:00'
image: /images/mcmc_chain_not_converge.png
title: Fixing sampler errors in probit regression with rstanarm
---

I was working through problem 15.5 of [*Regression and Other Stories*](https://avehtari.github.io/ROS-Examples/), which asks to fit a probit regression to a previous example with a logistic regression.
I used a model I had built on the [National Election Survey dataset](https://github.com/avehtari/ROS-Examples/tree/master/NES) (on `rstanarm` 2.21.1):

```R
fit_nes_probit <-
 rstanarm::stan_glm(rvote ~ income_int_std + gender + race +
                           region + religion + education_cts +
                           advanced_degree + party + ideology3 +
                           gender : party,
        family=binomial(link="probit"),
        data=nes92)
```

When I got this error about the chains not converging:

```
SAMPLING FOR MODEL 'bernoulli' NOW (CHAIN 1).
Chain 1: Rejecting initial value:
Chain 1:   Log probability evaluates to log(0), i.e. negative infinity.
Chain 1:   Stan can't start sampling from this initial value.
...
Chain 1:
Chain 1: Initialization between (-2, 2) failed after 100 attempts.
Chain 1:  Try specifying initial values, reducing ranges of constrained values, or reparameterizing the model.
[1] "Error in sampler$call_sampler(args_list[[i]]) : Initialization failed."
error occurred during calling the sampler; sampling not done
Error in check_stanfit(stanfit) :
  Invalid stanfit object produced please report bug
```

Even worse it crashed my Jupyter IRkernel, and the only way I could get it running again was to restart the whole process!
I continued to debug it in an R console which wouldn't crash after a fail.

This surprised me because I found it worked fine with `link="logit"`, that is logistic regression.
The solution was to set a lower scale value in the prior (the default is 4):

```R
fit_nes_probit <-
 rstanarm::stan_glm(rvote ~ income_int_std + gender + race +
                           region + religion + education_cts +
                           advanced_degree + party + ideology3 +
                           gender : party,
        prior=rstanarm::normal(scale=0.5, autoscale=TRUE),
        family=binomial(link="probit"),
        data=nes92)
```

# Why this works

To work out what was going on I started trying to simulate similar data that led to the same error.
After some experimentation I found a simple reproducable failing example with a few noise predictors:

```R
N <- 1000
fake_data <- data.frame(a=rbinom(N,1,0.5),
                        b=rbinom(N,1,0.5),
                        c=rbinom(N,1,0.5),
                        d=rbinom(N,1,0.5),
                        e=rbinom(N,1,0.5),
                        y=rbinom(N,1,0.5))

fit <- rstanarm::stan_glm(y ~ a + b + c + d + e
                          family=binomial(link="probit"),
                          data=fake_data)
```

This seems to have a better than 25% chance of failing because one of the chains fails most times I run it.
If I remove just one of the predictors it succeeds.

I got suspicious about the priors, in *Regression and Other Stories* they state (at least for logistic regression) the priors on the coefficients are `2.5/sd(x)`.
Here the standard deviation of all our predictors is just 0.5 (since they're binomial with probaibility 0.5).
I tried to check this on a model that fit:

```R
fit <- rstanarm::stan_glm(y ~ a + b + c + d,
                          family=binomial(link="probit"),
                          data=fake_data)
prior_summary(fit)
```

Which resulted in:

```
Priors for model 'fit'
------
Intercept (after predictors centered)
 ~ normal(location = 0, scale = 4)

Coefficients
  Specified prior:
    ~ normal(location = [0,0,0,...], scale = [4,4,4,...])
  Adjusted prior:
    ~ normal(location = [0,0,0,...], scale = [7.98,8.00,7.97,...])
------
```

So it looks like the default scale was 4, and they've been increased by the inverse standard deviation, that is a factor of 2.

The next thing I tried was a flat prior on the coefficients, which converged (to my surprise):

```R
fit_flat <- rstanarm::stan_glm(y ~ a + b + c + d + e,
                          family=binomial(link="probit"),
                          prior=NULL,
                          data=fake_data)
```

The next idea I had was to make the priors slightly tighter, to try to make the coefficients closer to 0.
Some experimentation found it would start to converse consistently around a scale of 1:

```R
fit_weak <- rstanarm::stan_glm(y ~ a + b + c + d + e,
                          family=binomial(link="probit"),
                          prior=rstanarm::normal(scale=1, autoscale=TRUE),
                          data=fake_data)
```

I found for the NES model I was fitting I had to further reduce the scale to 0.5 for it to converge, or alternatively set a flat prior.

I don’t understand how Stan works, but I’m guessing as we add more predictors there’s a greater chance of getting one that is far from it’s actual value (in this case 0).
I suspect because the probabilities in the normal distribution go to zero much faster than in the logistic distribution, this leads to numerical errors in `link="probit"` that don’t occur in `link="logit"`.
I've asked in the [Stan Discourse](https://discourse.mc-stan.org/t/default-prior-for-probit-regression-with-many-predictors-fails-to-sample/23959) to see if anyone can give more insight.

# Getting the data

For reproducability, here's how the `nes92` dataset above was generated, with a bit of feature engineering.

```R
library(dplyr)

nes <- foreign::read.dta('https://raw.githubusercontent.com/avehtari/ROS-Examples/master/NES/data/nes5200_processed_voters_realideo.dta')

nes92 <-
nes %>%
filter(year == 1992) %>%
# Only people who actually voted republican or democrat
filter(!is.na(presvote_2party)) %>%
transmute(
    age= age,
    gender=gender,
    race=race,
    region=region,
    income=income,
    religion=religion,
    education=educ3,
    party = partyid3_b,
    rvote = presvote_2party == '2. republican',
    ideology = ideo7, # ideo is just a summary
) %>%
select(-father_party, -mother_party) %>%
filter_all(~!is.na(.))  %>%
mutate(income_int = as.integer(income) - 1,
       income_int_std = (income_int - mean(income_int))/(2*sd(income_int)),
       advanced_degree = education == '7. advanced degrees incl. llb',
       education_cts = (as.integer(education) - 5)/6,
       ideology_int = (as.numeric(ideology) - 5)/6,
       ideology3 = if_else(ideology_int < 0, "liberal", if_else(ideology_int > 0, "conservative", "moderate")))
```
