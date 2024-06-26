---
date: 2010-11-27 06:48:27+00:00
image: /images/cas-maxima.png
title: The point of computer algebra systems
---

I wanted to do the contour integral of  $\frac{1}{z-a}$  around the unit circle on a computer for kicks. So I parameterised it with  $e^{it}$  and entered it into Maxima and Matlab


 $\int_0^{2\pi} \frac{i e^{it}}{e^{it}-a} dt$


for various values of  $a$ . By Cauchy’s integral formula we know it should be  $2 \pi i$  if  $|a|<1$  and 0 if  $|a|>1$ . Interestingly both programs gave the right answer for $a=0$ (I suppose the calculation is easy there) but gave totally wrong answers for  $a \neq 0$  (Matlab gave $0$ for  $|a|<1$  and  $-2 \pi i$  if  $|a|>1$  and Maxima gave $0$ everywhere).


I have no idea why this happens! In Matlab if you expand the complex exponential it gives the right answer. In Maxima it then has trouble computing the integral – but if you perform operations to make the denominator real and perform some simplifications it gives the right answer.


Now people sometimes ask *Why learn arithmetic when we have calculators?* I think this shows exactly why: you need to know when your calculator is giving you garbage.


<!--more-->


Calculators are pretty good – but you can still trick them. Taking an example out of Hjorth Jensens’ superb notes on Computational Physics [available [online](http://www.physics.ohio-state.edu/~ntg/780/)]


 $\frac{\sin x}{1+\cos x} = \frac{1-\cos x}{\sin x}$


However if I put these into my pocket calculator for  $x=7 \times 10^{-7}$  I get


 $3.5 \times 10^{-7} = 3.514285714 \times 10^{-7}$.


(This is of course just *round off error* due to the finite precision that a calculator works at). Understanding the ideas behind how your calculator does arithmetic lets you understand why it can screw up – another common error is the ordering of operations.


Computer Algebra Systems (which are just more sophisticated calculators) are useful for the same reason pocket calculators are: doing work that you understand how to do (and hopefully can check) but would be laborious for you to do.


For instance if I want to calculate the square root of $170$ to a given precision manually. E.g. I could use


 $\sqrt{170}=13\sqrt{1+\frac{1}{169}}$


and use a Taylor expansion, or use a recursive linearisation: for  $a \ll x$  we have  $(x+a)^2 \approx x^2 + 2ax$ , so we start with $x=13$ (which is close to a solution), solve for $a$, which gives us a new $x=13+a$ which we feed back into the algorithm. (Or one of the other [methods](http://en.wikipedia.org/wiki/Methods_of_computing_square_roots)). In any case it would require a large number of divisions and multiplications with an increasing number of digits, and it gets very laborious as the precision grows.
If I’m calculating lots of square roots to high precision this could take weeks (unless I bought a good slide rule). Calculators are fantastic time savers.


Similarly Computer Algebra Systems should only be used on things you know how to calculate but which would take considerable effort to do so. (Modern CAS will often not give answers or even give incorrect answers if you don’t set up the problem in the right way, so you have to be very careful). They can’t supplant understanding, but they can save mounds of time.


Postscript: Since I wrote this article I have found an error in Mathematica 8.0. Typing


```
Series[HarmonicNumber[n-1, 2], {n, Infinity, 5}]
```

gives the result


 $\frac{\pi ^2}{6}-\frac{1}{n}-\frac{1}{2 n^2}-\frac{1}{6 n^3}+\frac{1}{2  n^4}+\frac{31}{30 n^5}+\textrm{O}\left(\frac{1}{n^6}\right)$


but a bit of messing around with the Euler-Maclaurin formula gives


 $\sum_{k=1}^{n-1} \frac{1}{k^2}= \frac{\pi^2}{6} - \frac{1}{n} - \frac{1}{2n^2} -\frac{1}{6n^3} + \frac{1}{30n^5}+ \textrm{O}\left(\frac{1}{n^7}\right)$


These are clearly in disagreement, and not in an obvious way! The only ways you could find out the Mathematica result is wrong is to derive the formula independently by hand or check the results numerically (for this particular case it’s useful to note that the error in the approximation is less than the first discarded term – see Concrete Mathematics by Graham, Knuth and Patashnik).


This reinforces the moral of the story: Computer Algebra Systems can be very useful in making your mathematics more efficient (it can be tedious deriving Euler-Maclaurin formula by hand), but they are currently far from perfect and you have to be sceptical about its output. This means you really have to understand the mathematics behind what you’re doing and check the results that are important to you using more reliable tools (checking special cases, testing numerically, …). Ideally you understand a little bit about how your CAS works.


This is all of course equally true of complicated numerical solvers – e.g. differential equation solvers. You need to check it’s giving you sensible output, because you may just have chosen parameters that your particular algorithm is terrible at.