---
categories:
- stan
- r
- data
- bayesian
date: '2021-08-19T19:01:51+10:00'
image: /images/glm_priors.png
title: Stan Linear Priors
---

This is the second on a series of articles showing the basics of building models in Stan and accessing them in R.
In the [previous article](/getting-started-rstan) I showed how to specify a simple linear model with flat priors in Stan, and fit it in R with a formula syntax.
In this article we extend this to specify priors; defaulting to general weakly informative priors but allowing use of specific priors.

# Priors in linear regression

In our previous model $$ y \sim N(\alpha + \beta x, \sigma) $$, we didn't specify any priors for our parameters, the intercept $$\alpha$$, the coefficients $$\beta$$ and the residual standard deviation $$\sigma$$.
We can extend our Stan model to take data specifying these priors, and to declare the priors themselves in the model.

Following `rstanarm::stan_glm` it would be nice that if you didn't specify a prior for it to use a reasonable default prior.
Following the discussion in [*Regression and Other Stories*](https://avehtari.github.io/ROS-Examples/), Section 9.5, we can use the same weak priors that they use that keep inferences stable, but don't have much impact on the estimates.

The default prior for the coefficients is $$\beta \sim N(0, 2.5 s_y/s_x)$$.
Centring on 0 also makes sense without knowing which direction the coefficients should lie in.
The ratio of standard deviations is important for the prior to be invariant under rescaling transformation.
If we were to rescale $$y' = k y$$ and $$x' =  A x$$, then the coefficients would scale as $$\beta' = kA^{-1} \beta $$, so our prior should rescale in an analogous way.
The factor 2.5, quoting from *Regression and Other Stories*, "is somewhat arbitrary, chosen to provide some stability in estimation while having little influence on the coefficient estimate when data are even moderately informative".
Perhaps the worst part of this assumption is that as you add more coefficients (and especially interactions) that the prior stays the same and they are all independent.
Perhaps a better approach would be a joint distribution where some of the coefficients were more spread than others, since in many cases as you get more predictors a few of them may have a significant association but most will not.

The default weak prior for the intercept $$\alpha$$ is given indirectly by assigning a prior the expected value of y at the mean value of x is normally distributed with mean the mean value of y, and standard deviation 2.5 times the standard deviation of y; that is $$ E(y | x=\bar{x}) \sim N(\bar{y}, 2.5 s_y)$$.
Essentially we're saying that at the centre of x, the data should be near the centre of y, and the error scales with the standard deviation of y (using a similar rescaling argument as above), again picking 2.5 as a .
This is better than putting a prior directly on the intercept, because it's invariant in a translation of $x$ or $y$; we're always evaluating near the centre of the data (where we're likely to have the most information).
The expected value of y in our model is precisely $$ \alpha + \beta x$$, so we can rearrange this into  $$\alpha \sim N(\bar{y} - \beta \bar{x}, 2.5 s_y) $$.

Finally for the residual standard deviation assumed prior is $$ \sigma \sim {\rm Exponential}(1/s_y) $$.
This means in particular that the expected value is $$s_y$$, which is reasonable from scaling assumptions, and that the value is non-negative.
I'm not sure how reasonable the assumption in the distribution itself is, but I'll take it as a given.

We further want to be able to extend from these default priors to enable passing informative priors.
We can directly extend the prior for the coefficients to take a centre vector and standard deviation vector (or more generally a covariance matrix) that can be passed in place of the default priors.
For the intercept $$\alpha$$ we could similarly specify a centre point and standard deviation, but to conform with the weak prior form we could pretend the data is centred, $$\bar{x} = 0$$, so the $$\beta$$ coefficient has no influence on the prior.
Finally for the standard deviation we could pass a different parameter for the exponential distribution than the inverse standard deviation of y.

# Writing a Stan Model

With this plan we extend our Stan data to include the centre

$$ \begin{align}
\beta &\sim N(\mu_\beta, s_\beta) \\
\alpha &\sim N(\mu_\alpha - \beta \bar{x}, s_\alpha)\\
\sigma &\sim {\rm Exponential}(1/{\mu_\sigma})
\end{align} $$

Following `rstanarm` I refer to the centre as the `location` and the standard deviation as the `scale`, and I call the parameter in the exponential distribution the `rate`.
Note that the priors are specified as part of the model.


```stan
// Linear model - linear.stan
data {
  int<lower=0> N;       // Number of data points
  int<lower=0> K;       // Number of predictors
  matrix[N, K] X;       // Predictor matrix
  real y[N];            // Observations 

  // NEW: Data specifying priors
  vector[K] prior_location;                // Coefficient Normal Prior - centre
  vector[K] prior_scale;                   // Coefficient Normal Prior - standard deviation
  real prior_intercept_location;           // Intercept Normal Prior - centre
  matrix[1, K] prior_intercept_predictor;  // Intercept Normal Prior - offset centre by -beta * prior_intercept_predictor
  real prior_intercept_scale;              // Intercept Normal Prior - standard deviation
  real prior_aux_rate;                     // Exponential prior on sigma
}
parameters {
  real alpha;           // intercept
  vector[K] beta;       // coefficients for predictors
  real<lower=0> sigma;  // error scale
}
model {
  // NEW: Prior distributions
  beta ~ normal(prior_location, prior_scale);
  alpha ~ normal(prior_intercept_location - prior_intercept_predictor * beta, prior_intercept_scale);
  sigma ~ exponential(prior_aux_rate);
  
  // Target Density
  y ~ normal(alpha + X * beta, sigma); // target density
}
```

# Running the model from R

As before we can wrap this in a function, adding extra parameters for the priors.
Note I set the defaults to `FALSE`; it would have made more sense to use `NULL` but I had an idea that I could copy `rstanarm`'s approach of using `NULL` for a flat prior before realising I'd need to do a [lot of work](https://github.com/stan-dev/rstanarm/blob/master/src/stan_files/continuous.stan) to add that kind of flexibility.

One thing that caught me is a vector of length 1 will be treated as a scalar, not a vector, by Stan (because it's hard to distinguish these in R), and so we need to wrap prior vectors passed to RStan in `array`.
From the [RStan vignette](https://cran.r-project.org/web/packages/rstan/vignettes/rstan.html)

> If we want to prevent RStan from treating the input data for y as a scalar when N‘ is 1, we need to explicitly make it an array

```R
fit_stan_linear <- function(formula, data,
                            ...,
                            prior_location=FALSE,
                            prior_scale=FALSE,
                            prior_intercept_location=FALSE,
                            prior_intercept_scale=FALSE,
                            prior_aux_rate=FALSE) {
    y <- model.response(model.frame(formula, data))
    X <- remove_intercept_from_model(model.matrix(formula, data))
    
    K <- ncol(X)
    N <- nrow(data)
    
    if (isFALSE(prior_location)) {
        prior_location <- rep(0, K)
    }
    
    if (isFALSE(prior_scale)) {
       prior_scale <-  2.5 * sd(y) / apply(X, 2, sd)
    }
    
     if (isFALSE(prior_intercept_scale)) {
       prior_intercept_scale <- 2.5 * sd(y)
    }
    
    if (isFALSE(prior_aux_rate)) {
       prior_aux_rate <- 1/sd(y)
    }
    
    if (isFALSE(prior_intercept_location)) {
        prior_intercept_location <- mean(y)
        prior_intercept_predictor <- matrix(apply(X, 2, mean), ncol=K)
    } else {
        # When a specific location is set, remove the effect of predictor offset
        # by setting it to 0
        prior_intercept_location <- prior_intercept_location
        prior_intercept_predictor <- matrix(rep(0,K), ncol=K)
    }
    
    
    fit <- rstan::stan(
        file = "linear.stan", 
        data = list(
            N = nrow(X),
            K = ncol(X),
            X = X,
            y = y,
            prior_intercept_predictor = prior_intercept_predictor,
            prior_intercept_location = prior_intercept_location,
            # Need array when there is just 1 predictor
            prior_scale = array(prior_scale, dim=K),
            prior_location = array(prior_location, dim=K),
            prior_centre_scale = prior_intercept_scale,
            prior_sigma_rate = prior_aux_rate
          ),
        ...
        )
    
    names(fit) <- get_linear_names(names(fit), colnames(X))
    
    structure(list(fit=fit, formula=formula, data=data), class=c("my_linstan"))
}
```

# Testing using priors

As a test of this functionality let's compare `rstanarm::stan_glm` with our function on the [SexRatio](https://github.com/avehtari/ROS-Examples/tree/master/SexRatio/) data from Section 9.5 of *Regression and Other Stories* (inspired by a study of the effect of Beauty on the sex ratio of children, where there is weak data and small priors).

We have a small data set of 5 points, representing the percentage of girl babies $$y$$, as a function of standardised beauty $$x$$.


```R
x <- seq(-2,2,1)
y <- c(50, 44, 50, 47, 56)
sexratio <- data.frame(x, y)
```

The weakly informative priors give similar results to the minimum likelihood estimator found by `lm`.

```R
fit_sexratio_lm <- lm(y~x, data=sexratio)
fit_sexratio_default <- stan_glm(y ~ x, data=sexratio)
fit_sexratio_default_stan <- fit_stan_linear(y ~ x, data=sexratio)
```

The coefficients are all very close to this (the estimated residual deviation of `lm` is a little lower at 4.3).

```
            Median MAD_SD
(Intercept) 49.3    1.9
x            1.4    1.4

Auxiliary parameter(s):
      Median MAD_SD
sigma 4.6    1.7
```

However we can add an informative prior (which [acts as regularisation](/prior-regularise) on the coefficients), based on the fact the rate of girl births is around 48.5% to 49%, and based on prior studies we wouldn't expect beauty to have more than a 0.8 percentage point impact on the rate of girl births.


```R
fit_sexratio_post <- stan_glm(y ~ x, data=sexratio, prior=normal(0,0.2), prior_intercept=normal(48.8, 0.5))
fit_sexratio_post_stan <- fit_stan_linear(y ~ x, data=sexratio,
                                          prior_location=0,
                                          prior_scale=0.2,
                                          prior_intercept_location=48.8,
                                          prior_intercept_scale=0.5)
```

These give identical coefficient estimates to one decimal place.

```
            Median MAD_SD
(Intercept) 48.8    0.5
x            0.0    0.2

Auxiliary parameter(s):
      Median MAD_SD
sigma 4.3    1.3
```

Now we know how to fit a simple linear model in Stan and add priors, it would be nice if we could make predictions and take posterior draws from it.
That's covered in the next article [making Bayesian predictions with Stan and R](/rstan-predictions).