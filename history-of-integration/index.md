---
date: 2011-11-06 06:01:22+00:00
image: /images/smith-volterra-cantor_set.svg
title: Some history of integration
---

This post is based mainly on a chapter in *A Radical Approach to Lebesgue’s Theory of Integration* by David Bressoud in which he explores the history of the Lebesgue integral. The story I will tell is closer to folklore than a historical account, but nevertheless enlightening.


<!--more-->The modern concept of a limit, central to how we understand analysis today, was not formulated until (arguably) 1821 by Cauchy, despite calculus being invented in the late 17th century and limiting approaches extending back into Greek mathematics. A major reason for the time it took for the rigorous foundations of analysis to develop is because they were not really necessary – most manipulations in mathematics and science involved “well-behaved” analytic functions. (This was well before the notion of a set was in vogue, so a function was generally considered to be a “formula” or a geometric concept, not an arbitrary mapping of one set into another). However there were exceptions to this; for instance trigonometric series, used by Bernoulli to solve a vibrating string problem in 1753 and by Fourier in 1821 to solve a heat equation.


Trigonometric (or Fourier) series were useful for solving a wide range of physical problems, and at the same time challenged mathematical intuition, for instance Fourier found  $\frac{4}{\pi}\sum_{n=1}^{\infty} \frac{(-1)^{n-1}}{2n-1} \cos(\frac{(2n-1)\pi x}{2})$  converged to a square wave – a discontinuous function.


In investigating these sorts of pathological functions rigorous notions of continuity, differentiability, uniform convergence and integrability arose. In particular Riemann’s definition of an integral (I can’t find his original definition, so this is a modern version)


A *partition* of an interval  $[a,b]$  is a sequence of points  $a=a_0 < a_1 < \ldots < a_n=b$ . The *mesh* of such a partition is the maximum of  $a_{i}-a_{i-1}$  for  $i=1,\ldots,n$ . A *tagging* of such a partition is points  $\{t_1,\ldots,t_n\}$  satisfying  $a_{i-1} \leq t_i \leq a_i$  for  $i=1,\ldots,n$ .


A function  $f$  on  $[a,b]$  is integrable with integral  $A$  if and only if for every positive quantity  $\epsilon$  there is a positive quantity  $\delta$  such that for any tagged partition  $\{a_0,\ldots,a_n\},\{t_1,\ldots,t_n\}$  with mesh less than  $\delta$ ,  $\left|A - \sum_{j=1}^{n} f(t_j) (a_j - a_{j-1})\right| < \epsilon$ .


Informally the integral of a positive function is the area under its graph. The idea behind Riemann’s definition is to approximate the area by adding together rectangles, with the length given by a partition and height the value of the function at the tag. Riemann’s definition essentially says that if, as the width of these rectangles approaches zero, their sum approaches a constant number independent of the way we choose the rectangles, this must be the area under the graph.


There are ‘problems’ with Riemann’s definition though, one of which is that not every derivative is integrable. To demonstrate this we construct [Volterra’s function](http://en.wikipedia.org/wiki/Volterra_function).


We begin by constructing a Smith-Volterra-Cantor set (a.k.a a Fat Cantor set). Start (step 0) with the interval [0,1] and inductively at step n remove an open set of length  $1/4^n$  from each of the  $2^{n-1}$  connected subsets. The intersection of all these sets forms a nowhere-dense set. Its ‘outer measure’ is  $1 - (1/4 + 2/16 + 4/64 + \ldots) = 1/2$ .


I won’t explicitly detail the construction of the Volterra function. It uses the function  $x^2\sin(1/x)$  which is differentiable, but the derivative is not continuous at 0. The Volterra function then uses this function to construct a function that is differentiable, but the derivative is not continuous on the Smith-Volterra-Cantor set and so (by Lebegue’s criterion for Riemann integrability) isn’t Riemann integrable.


So it’s worth looking for a better integral. Since integration corresponds to finding the area of a graph, one method is to try to assign a size to sets in the plane. But even assigning lengths to subsets of the line is difficult.


Peano and Jordan assigned a size to sets using intervals (or in 2 dimensions, rectangles) using a method similar to the “proof by exhaustion” used by the Greeks. The idea behind proof by exhaustion is to prove the area of an object is A by proving it is not less than A and proving that it is not more than A. The *inner content* of a set is the supremum of the finite sums of lengths of intervals with non-intersecting interiors that are contained by the set. The *outer content* of a set is the infimum of the finite sums of lengths of intervals with non-intersecting interiors that contain the set. A set is *Jordan measurable* if its inner content is equal to its outer content and this value is called the *Jordan measure*.


Unfortunately the Smith-Volterra-Cantor set is not Jordan measurable: it contains no intervals so its inner content is 0, but its outer content is 1/2.


Borel took a different approach, by defining the lengths of countable disjoint unions of intervals to be the sums of the lengths of the intervals, and the length of $B \setminus A$ to be the length of B minus the length of A. Consequently the Smith-Volterra-Cantor set is measurable.


However the cardinality of the Borel measurable sets is  $\mathfrak{c}$  (the cardinality of the reals) since every Borel measurable set can be constructed from the intervals (which have cardinality  $\mathfrak{c}$ ) by complementation and countable unions. To contrast the Cantor set has outer content zero, so every subset will have outer content zero, and hence be Jordan measurable. Since the Cantor set is uncountable this implies the cardinality of the Jordan measurable sets is at least  $2^\mathfrak{c}$ .


Lebesgue’s criterion for a subset of [a,b] to be measurable is a subtle play on Borel’s and Peano-Jordan’s. The *outer measure* of a set S,  $m^\star(S)$ , is the sum of the infimum of the sum of **countably** many intervals which have a union containing the set. The *inner content* of a set is the length of [a,b] minus the outer measure of the sets complement in [a,b], that is  $b-a-m^*([a,b]\setminus S)$ . A subset of [a,b] is *Lebesgue measurable* if its inner measure and outer measure are equal. A subset of the real line is Lebesgue measurable if its intersection with [-n,n] is for every positive integer n. Carathéodory came up with the equivalent criterion: a set E is Lebesgue measurable if  $m^*(A) = m^*(A \cap E) + m^*(A\setminus E)$  for all sets A.


This final result is normally what is presented early in a course on Lebesgue integration, but I hope it seems a little less mysterious now. The countable unions and insisting  $m^*(A \setminus E) = m^*(A) - m^*(E)$  if A contains E is essential to measure sets such as the Smith-Volterra-Cantor set. Using the outer measure ensures we automatically get the non-Borel sets of measure zero (and in fact a subset of the real line is Lebesgue measurable if and only if it is the disjoint union of a Borel measurable set and a set of measure zero).


The Lebesgue integral is great; it can integrate a much larger suite of functions than the Riemann integral, it has strong convergence theorems (dominated convergence theorem, Fubini’s theorem) and it readily abstracts (and forms the basis for modern probability theory). However it still can’t integrate every derivative: consider the sinc function  $\frac{\sin x}{x}$ , it is the derivative of the function defined on the whole real line with Taylor expansion  $\sum_{n=0}^{\infty} \frac{(-1)^n}{(2n+1)(2n+1)!}x^{2n+1}$ , but it is not Lebesgue integrable over the whole real line. The problem is the integral oscillates too fast; the total area above the x-axis is infinite and the total area below the x-axis is infinite, but if you add them from the origin out they cancel to a finite sum (it is the limit of Riemann integrals). (There are examples on bounded intervals too).


There is an integral more powerful than the Lebesgue integral on the real line, the Henstock-Kurzweil-Denjoy-Perron integral, or as it is sometimes known, the generalised Riemann integral, defined on a bounded interval as follows:


The function f is integrable on [a,b] with integral A if for every positive number  $\epsilon$  there exists a positive function  $\delta$  on [a,b] such that every tagged partition $\{a_0,\ldots,a_n\},\{t_1,\ldots,t_n\}$  satisfying  $t_i - \delta (t_i) \leq a_{i-1} \leq t_i \leq a_i \leq t_i + \delta (t_i)$ ,  $\left|A - \sum_{j=1}^{n} f(t_j) (a_j - a_{j-1})\right| < \epsilon$ .


This integral (once extended to the whole real line) contains every Lebesgue integrable function, limits of Lebesgue integrable functions and some new functions to boot! In particular every derivative is integrable. (Robert Bartle’s *A modern theory of integration* gives an accessible exposition on the subject).


Of course there’s more: Cesaro-Denjoy integrals, approximate Perron integrals and generalisations to higher dimensions (where finding an integral for which ‘every derivative’ is integrable is, to my knowledge, unsolved).


I’d like to conclude by reflecting how solving problems in physics by questionably performing operations on infinite trigonometric series was a major source of inspiration for mathematics, and I wonder how much impact resolving the questionable path-integrals in Quantum Field Theory will have (and has already had) on mathematics.
