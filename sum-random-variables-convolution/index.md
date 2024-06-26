---
categories:
- data
- statistics
date: '2021-04-09T08:00:00+10:00'
image: /images/Convolution_Cross_Multiplication.jpg
title: How to Sum Random Variables
---

Suppose you've got two dice; what's the probability the sum of their rolls will add up to 4?
You simply look at all the ways the values of the dice could sum to 4 (e.g. 1 and 3, 2 and 2 or 1 and 4), and add up their probabilities (in this case each is 1/36, so totalling 3/36 or 1/12).
You could use a [probability square](/probability-square) to visualise this calculation.

This is the general procedure to sum two independent random variables.
You've got two random variables X and Y; in the example above the dice.
They each have some probability distribution function $f_X$ and $f_Y$ which encode how probable any particular outcome is.
For the dice the probability distribution function is just $f_X(x) = \frac{1}{6},\, x \in \{1,2,3,4,5,6\}$ (that is the probability of any number between 1 and 6 is one-sixth).
Then the probability distribution of their sum is then $f_{X+Y}(x) = \sum_{y+z=x} f_X(y) f_Y(z)$; that is for each pair of outcomes that sum up to x we multiply the probability of them occurring.
In our dice example we calculated $f_{X+Y}(4) = \frac{1}{12}$; we could repeat the calcuation for all numbers between 2 and 12 to get the full probability distribution function.

The sum above can be rewritten in a slightly different form using a change of variables; $f_{X+Y}(x) = \sum_{y} f_X(y) f_Y (x - y) =: (f_X * f_Y)(x)$.
This last expression is just noting this is the definition of a [convolution](https://en.wikipedia.org/wiki/Convolution).
A shorthand way of writing this is $f_{X+Y} = f_X * f_Y$ (note that this is convolution and *not* multiplication!).

The [convolution theorem](https://en.wikipedia.org/wiki/Convolution_theorem) says that the Fourier Transform maps convolutions into products.
So in particular we can rewrite $f_{X+Y} = \mathcal{F}^{-1} (\mathcal{F}(f_X) \mathcal{F}(f_Y))$
This can be computationally convenient; a fast fourier transform can be more efficient than performing the sum in the convolution manually.
To see this in action we can encode our dice example in R; we represent our probability density function as a vector which are the values of the function at 0, 1, 2, ...

```R
# Vectors are 0 indexed, so probability of 0 as 0
# Probability at 1-6 as 1/6
# We need extra space at the end for higher values so we pad the end with 0s
x <- c(0, seq(1/6, 6), seq(0, 10))
y <- x

# Convoluiton using the fourier transforms and their inverse
z <- fft(fft(x) * fft(y), inverse=True)
# Remove the imaginary part (which should be 0)
z <- abs(z)
# Renormalise the probability to sum to 1
# (The fft introduces a constant multiplicative factor)
z <- z / sum(z)

# The result is what we would get from the convolution
# For example the value at 4 (the 5th element in the vector) is 3
print(round(z*36, 0.01))
# [1] 0 0 1 2 3 4 5 6 5 4 3 2 1 0 0 0 0
```

These relations are also useful for theoretical calculations.
For example the PDF of a standard normal distribution, $\frac{1}{\sqrt{2\pi}}e^{-\frac{x^2}{2}}$ is a fixpoint of the Fourier transform.
This with the [scaling and shift relations](https://en.wikipedia.org/wiki/Fourier_transform#Tables_of_important_Fourier_transforms) make it easy to calculate the [sum of normally distributed variables](https://en.wikipedia.org/wiki/Sum_of_normally_distributed_random_variables#Using_the_convolution_theorem).
In particular if $X \sim \mathcal{N}(\mu_1, \sigma_1^2)$ and $Y \sim \mathcal{N}(\mu_2, \sigma_2^2)$  then $X + Y \sim \mathcal{N}(\mu_1 + \mu_2, \sigma_1^2 + \sigma_2^2)$.