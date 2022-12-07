---
categories:
- ''
date: '2021-09-20T23:34:31+10:00'
draft: true
image: /images/
title: Hierarchical Binomial
---

$$\{\gamma_{i,s}\}_{i=1}^{n_s} \sim \theta_s \quad \forall s=1,\ldots,m$$

$$\theta_s \sim {\rm Beta}(\omega \tau + 1, (1-\omega) \tau + 1)$$

$$\begin{align}
{\mathbb P}\left( \{\theta_s\}_{s=1}^{m}, \omega, \tau | \{\gamma_{i,s}\}\right) &= \frac{{\mathbb P}\left(\{\gamma_{i,s}\} \vert \{\theta_s\}_{s=1}^{m}, \omega, \tau \right) {\mathbb P}(\{\theta_s\}_{s=1}^{m}, \omega, \tau)}{{\mathbb P}(\{\gamma_{i,s}\})} \\
&= \frac{\left[\prod_{s=1}^{m}{\mathbb P}\left(\{\gamma_{i,s}\} \vert \theta_s\right) \right] \left[\prod_{s=1}^{m} {\mathbb P}(\theta_s \vert \omega, \tau)\right] {\mathbb P}(\omega, \tau)}{\int \left[\prod_{s=1}^{m}{\mathbb P}\left(\{\gamma_{i,s}\} \vert \theta_s\right) \right] \left[\prod_{s=1}^{m} {\mathbb P}(\theta_s \vert \omega, \tau)\right] {\mathbb P}(\omega, \tau) \, {\rm d}\theta_1 \cdots {\rm d}\theta_m {\rm d} \omega {\rm d} \tau} \\
&= c {\mathbb P}(\omega, \tau) \prod_{s=1}^{m} \theta^{N_s p_s + \omega \tau} (1 - \theta)^{N_s(1-p_s) + (1-\omega)\tau}
\end{align}
$$

where c is the normalising constant so the probability integrates to 1.
In particular:

$$\begin{align}
c^{-1} &= {\int \left[\prod_{s=1}^{m}{\mathbb P}\left(\{\gamma_{i,s}\} \vert \theta_s\right) \right] \left[\prod_{s=1}^{m} {\mathbb P}(\theta_s \vert \omega, \tau)\right] {\mathbb P}(\omega, \tau) \, {\rm d}\theta_1 \cdots {\rm d}\theta_m {\rm d} \omega {\rm d} \tau} \\
&= \int_{-2}^{\infty} {\rm d}\tau \int_0^1 {\rm d}\omega \; {\mathbb P}(\omega, \tau) \prod_{s=1}^{m} \int_0^1 {\rm d}\theta_s \; \theta^{N_s p_s + \omega \tau} (1 - \theta)^{N_s(1-p_s) + (1-\omega)\tau} \\
& = \int_{-2}^{\infty} {\rm d}\tau \int_0^1 {\rm d}\omega \; {\mathbb P}(\omega, \tau) \prod_{s=1}^{m} B(N_s p_s + \tau \omega + 1, N_s (1-p_s) + \tau (1-\omega) + 1)
\end{align}
$$

where B is the [beta function](/beta-function).
I don't know a simple way to calculate these integrals for any non-trivial priors, but since they are only two dimensional they can be efficiently integrated with adaptive quadrature.
For example using a flat prior on $\omega$ and an exponential prior with rate 1 on $\tau$ (so ${\mathbb P}(\tau=t) \propto e^{-t},\; \forall t \in [0,\infty]$) in R:

```R
# The data are encoded as a list with one outcome per group
# Replace this with actual data
n <- c(9, 12)
z <- c(3,5)
p <- z/n

# Prior: Replace this with preferred prio
prior_fn <- function(omega, tau) exp(-tau) 

# Range of the variables
range_omega <- c(0,1)
range_tau <- c(0, Inf)


normalisation_function <- function(x) {
    omega <- x[1]
    tau <- x[2]
    a <- z + tau * omega + 1
    b <- n - z + tau * (1-omega) + 1
    exp(log_prior_fn(omega, tau) + sum(lbeta(a, b)))
}

# Normalisation constant evaluate by integration
# Requires the cubature package
normalisation <- cubature::hcubature(
                    normalisation_function, 
                    lowerLimit=c(range_omega[1], range_tau[1]), 
                    upperLimit=c(range_omega[2], range_tau[2]))



# Posterior
posterior <- function(omega, tau, theta) {
    a <- z + tau * omega + 1
    b <- n - z + tau * (1-omega) + 1
    likelihood <- prod(theta^(a-1)*(1-theta)^(b-1)) 
    prior_fn(omega, tau) * likelihood / normalisation$integral
}

# Check posterior integrates to 1
# cubature::hcubature(function (x) posterior(x[1], x[2], c(x[3], x[4])),
#                     c(range_omega[1], range_tau[1], 0, 0) ,
#                     c(range_omega[2], range_tau[2], 1, 1))
```

Note that when there is a large number of parameters the products are likely to have rounding issues, and it's better to add the log likelihoods then exponentiate.

Similarly using the Beta function marginals and expectation values require at most a two dimensional integral over $\omega$ and $\tau$.
For example the marginal posterior in $\omega$ and $\tau$ is:

```R
posterior_omega_tau <- function(omega, tau) {
    a <- z + tau * omega + 1
    b <- n - z + tau * (1-omega) + 1
    exp(log_prior_fn(omega, tau) + sum(lbeta(a, b)) - log(normalisation$integral))
}
```

Or to get a posterior of one of the $\theta_i$

```R
posterior_theta <- function(i, thetai) {
    cubature::hcubature(function(x) {
        omega <- x[1]
        tau <- x[2]
        a <- (z + tau * omega + 1)
        b <- (n - z + tau * (1-omega) + 1)
        dbeta(thetai, a[i], b[i])  *
        exp(log_prior_fn(omega, tau) + 
            sum(lbeta(a, b)) -
            log(normalisation$integral))
        }, c(0, 0), c(1, Inf))$integral
}

# Check normalisation: this should be very close to 1
# integrate(function(x) purrr::map_dbl(x, function(x) posterior_theta(1, x)), 0, 1)
```