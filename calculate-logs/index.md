---
categories:
- maths
date: '2020-09-09T08:00:00+10:00'
image: /images/slide_rule.jpg
title: Calculating Logs
---

Logarithms are a handy calculational tool for turning multiplication into addition.
You don't need any special tools to calculate approximate logarithms by hand, you just need to remember a few identities.

An easy example of how this works is with the identity $2^{10} = 1024 \approx 1000 = 10^3$.
Using base 10 logarithms this means that $10 \log 2 \approx 3$, or $\log 2 \approx 0.3$
We can then see because $2 \times 5 = 10$ that $log(2) + log(5) = 1$, and so $log(5) \approx 0.7$.
And then $7^2 = 49 \approx 50 = 5 \times 10$ and so $\log(7) \approx \frac{log(5) + log(10)}{2}$.
Using these kinds of rules we can quickly generate a table of logarithms.


| Number | Log  | Identity                      |
|--------|------|-------------------------------|
| 2      | 0.3  | $2^{10} \approx 10 ^ 3$   |
| 4      | 0.6  | $2^2 = 4$                 |
| 8      | 0.9  | $2^3 = 8$                 |
| 9      | 0.95 | $9^2 \approx 8 \times 10$ |
| 3      | 0.48 | $3^2 = 9$                 |
| 6      | 0.78 | $2 \times 3 = 6$                 |
| 5      | 0.7  | $2 \times 5 = 10$         |
| 7      | 0.85 | $7^2 \approx 5 \times 10$ |
| 11     | 1.05 | $11 \times 9 \approx 100$ |
| e      | 0.43 | $e^3 \approx 20$          |

So for example suppose we want to calculate $9 ! = 9 \times 8 \times 7 \times 6 \times 5 \times 4 \times 3 \times 2$.
The log is approximately the sum of the corresponding values in the log column; with is 5.56.
So the answer is around $10^{5.56}$, which looking in the table is around $4 \times 10^5$; this is within 10% of the correct answer.

Another way to do this is with [Stirling's approximation](https://en.wikipedia.org/wiki/Stirling%27s_approximation) $log 9! \approx 9 (\log 9 - \log e)  + \frac{1}{2} \log(2 \pi \times 9)$.
The first factor is $9 \times 0.52 \approx 4.7$, and approximating pi as 3 the second factor is $\frac{0.3 + 0.48 + 0.95}{2} \approx 0.87$.
So $9! \approx 10 ^ {5.6} \approx 4 \times 10 ^ 5$.

In my mind the most remarkable formula is $e^3 \approx 20.086$.
That this number is so close to 20 is a handy coincidence that makes it easy to convert between natural logarithms and logarithms base 10.
This is useful because we're familiar with base 10 exponents, but exponentials and natural logarithms have nice Taylor series for using in approximations.
So since $10 \ln(2) \approx 3 \ln(10)$ and $3 \approx \ln(20) \approx 1.3 \ln(10)$, we then get $\ln(10) \approx 2.3$.

These simple formulas make calculating products, decimals and roots to one decimal place really easy to calculate by hand or even in the head.
It's also easy to expand these further; composite numbers can be calculated by adding the logs of their factors, prime numbers by generating similar relations like $13 \times 7 = 91 \approx 9 \times 10$, $17 \times 7 = 119 \approx 4 \times 3 \times 10$, and $19 \times 11 = 209 \approx 3 \times 7 \times 10$.