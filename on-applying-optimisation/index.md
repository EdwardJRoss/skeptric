---
categories:
- maths
date: '2021-10-26T22:04:38+11:00'
image: /images/branch-and-bound.png
title: On Applying Optimisation
---

I recently watched the ACEMS public lecture [Optimal Decision Making: A tribute to female ingenuity](https://www.youtube.com/watch?v=jdcCtOp80jY), where [Alison Harcourt](https://en.wikipedia.org/wiki/Alison_Harcourt) (née Doig) talked about her contribution developing Branch and Bound ([William Cook has a paper on this history](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.437.5700&rep=rep1&type=pdf), and the landmark paper was [An Automatic Method of Solving Discrete Programming Problems (Land and Doig, 1960)](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.308.7332&rep=rep1&type=pdf)).
They talked about how impactful the work was on industry, particularly with logistics where you often have constrained resources (many indivisible) that you need to optimise some goal.
It's even useful for in Machine Learning for things like [Best Subset Selection](https://arxiv.org/abs/1507.03133) and [Optimal Sparse Decision Trees](https://arxiv.org/abs/1904.12847).

These techniques can solve the linear constrained problem

$$\begin{align}
& \text{maximize}   && \mathbf{c}^\mathrm{T} \mathbf{x}\\
& \text{subject to} && A \mathbf{x} \le \mathbf{b}, \\
&  && \mathbf{x} \ge \mathbf{0}, \\
\end{align} 
$$

where some of the x must be integers (the quadratic version is solvable too).
However it's hard to immediately see how this helps with real problems.

The optimisation part is straightforward; we're trying to maximise some objective (e.g. profit, negative cost, negative time).
The constraints then form minimum requirements and encode contention between resources or limits.
A classic example is the [Stigler Diet](https://en.wikipedia.org/wiki/Stigler_diet), trying to minimise the cost of a diet (objective) from a certain list of foods (variables) that meets all nutritional needs (constraints).

Putting something in this form can be difficult, but there are lots of benefits if you can.
In the 60 years since there's been improvements in the algorithms, computers are exponentially more powerful and there now exists very good software to run it.
Even with *lots* of variables and constraints the optimal solutions can be found relatively quickly, and can be recomputed as the information changes.

I haven't used these techniques much but I would love to use them more.
Part of the problem is I work mainly in digital spaces that have fewer constraints (although there are some, such as attention, screen space, computer time) and the problems I've dealt with often have only a few variables so can be solved with brute force methods.

The really hard thing seems like formulating the right objective and constraints (as in general defining the problem can harder than solving it).
The book [Model Building in Mathematical Programming by Williams](https://www.wiley.com/en-au/Model+Building+in+Mathematical+Programming%2C+5th+Edition-p-9781118443330) looks like a promising work in framing problems in this way.