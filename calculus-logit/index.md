---
categories:
- maths
date: '2021-08-08T08:00:00+10:00'
image: /images/scipy-special-expit-1.png
title: Calculus of the Inverse Logit Function
---

I was recently doing some logistic regression, and calculated the derivative of the Inverse Logit function (sometimes known as expit), to understand how the coefficients impact changes depending on the predicted probability.
It turns out it has some mathematically interesting properties that I thought would be fun to explore.

The inverse logit function is ${\rm logit}^{-1}(x) = \frac{\exp(x)}{1+\exp{x}}$.
A bit of calculus shows that

$$\frac{\rm d}{{\rm d} x} {\rm invlogit}(x) = \frac{e^{x}}{\left(1+e^{x}\right)^2} = {\rm invlogit}(x) (1 - {\rm invlogit}(x))$$

This is interesting in that if the predicted probability is p, then a small change in a predictor with a coefficient a should change the probability by approximately $a p (1-p)$.
This is maximised at $p=1/2$, where the local change in probability is $a/4$ which is the source of the divide-by-four rule in interpreting coefficients in logistic regression.

However I find this expression interesting and wanted to find out whether it *defines* the inverse logit function.
We want to find a function $f$ such that $f' = f(1-f)$.
Using the [derivative of the inverse function](https://en.wikipedia.org/wiki/Inverse_functions_and_differentiation) gives that 

$$\frac{\rm d}{{\rm d} x} f^{-1}(x) = \frac{1}{x(1-x)} = \frac{1}{x} + \frac{1}{1-x} \,.$$

Integrating gives $f^{-1}(x) = \log(x) - \log(1-x) + c = \log\left(\frac{x}{1-x}\right) + c$.
Up to an additive constant this is just the logit function.
Finally inverting this equation gives

$$f(x) = \frac{\exp(x-c)}{1 + \exp(x-c)} \,,$$

so that this indeed does define the inverse logit up to a translation.

Translating it to an inverse logit so that the maximum probability is at 0 gives it one more interesting property,

$$\begin{align} 1 - {\rm logit}^{-1}(x) &= 1 - \frac{\exp(x)}{1 + \exp(x)} \\ &= \frac{1}{1 + \exp(x)} \\ &= \frac{\exp(-x)}{1 + \exp(-x)} \\ &= {\rm logit}^{-1}(-x) \end{align}$$

Of course *this* symmetry property isn't defining, since any function defined on the positive numbers between 0 and 1 can be extended on the negative numbers to satisfy this property.