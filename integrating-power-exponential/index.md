---
categories:
- maths
date: '2021-03-02T18:32:51+11:00'
image: /images/integral_power_exponential.png
title: Integrating Powers of Exponentials
---

When working with distributions that were powers of exponentials, of which the normal and exponential distributions are special cases, I had to calculate the integrals of exponentials.
It's possible to transform these into expressions involving the Gamma function.
Specifically I found that for all positive p:

$$\int_{0}^{\infty} x^m e^{-x^p}\, \rm{d}x = \frac{\Gamma\left(\frac{m+1}{p}\right)}{p}$$

This is useful for calculating moments of powers of exponentials, namely for positive p and k:

$$\int_{-\infty}^{\infty} (x - c)^m e^{-\left\vert\frac{x-c}{k}\right\vert^p} \, \rm{d}x = \left\{\begin{array}{rl} 0, & \text{if }  m = 1,3,5,7,9,\ldots \\ \frac{2 k^{m+1} \Gamma\left(\frac{m+1}{p}\right)}{p}, & \text{if } m = 0, 2, 4, 6, 8, \ldots \end{array}\right.$$


This can all be proved with simple integration by substitution.
First starting with the moment integral $\int_{-\infty}^{\infty} (x - c)^m e^{-\left\vert\frac{x-c}{k}\right\vert^p} \, \rm{d}x$, where m is a non-negative integer and p and k are positive.
We can shift the integral so it centred on the midpoint c with the substitution $u = x - c$, which gives the integral $\int_{-\infty}^{\infty} x^m e^{-\left\vert\frac{x}{k}\right\vert^p} \, \rm{d}x$.

Now the term $e^{-\left\vert\frac{x}{k}\right\vert^p}$ is symmetric about the origin, and so when m is odd the integrand is anti-symmetric about the origin, and so the contributions for positive and negative x cancel out, resulting in a zero integral.
When m is even the integral is symmetric about the origin and so the contributions for positive and negative x are equal; so we can calculate the integral as $2 \int_{0}^{\infty} x^m e^{-\frac{x}{k}^p} \, \rm{d}x$, where we have dropped the absolute value because x/k is positive.

Finally we can remove k from the exponent by rescaling the distribution with the substitution $u = \frac{x}{k}$, giving the integral as $2 k^{m+1} \int_{0}^{\infty} x^m e^{-x^p} \, \rm{d}x$.
So we have reduced the problem to calculating the integral of a power of x multiplied by a power of the exponential.

The integral $\int_{0}^{\infty} x^m e^{-x^p} \,\rm{d}x$ can be transformed to remove the power from the exponent with the transformation $u = x^p$; this gives

$$\int_{0}^{\infty} x^m e^{-x^p} \,\rm{d}x  = \frac{1}{p} \int_{0}^{\infty} x ^{\frac{m + 1}{p} - 1} e^{-x} \,\rm{d}x$$ 

and by the definition of the Gamma function this is just $\frac{\Gamma\left(\frac{m+1}{p}\right)}{p}$.

It's easy to verify this in simple cases; when $p = 1$ we get the integral being $m!$ which can be proved by integration by parts.
For $m = 0$ and $p = 2$ this gives $\frac{\sqrt{\pi}}{2}$ which is [the Gaussian integral](https://en.wikipedia.org/wiki/Gaussian_integral).

These integrals are useful for calculating statistics of different exponential type distributions.