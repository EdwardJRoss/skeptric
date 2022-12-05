---
categories:
- maths
date: '2020-08-13T23:00:00+10:00'
feature_image_url: https://en.wikipedia.org/wiki/Analytic_continuation#/media/File:Imaginary_log_analytic_continuation.png
image: /images/imaginary_log_analytic_continuation.png
title: Complex Analysis
---

Imaginary numbers sound like a very impractical thing; surely we should only be interested in real numbers.
However imaginary numbers are very convenient for understanding phenomena with *real* numbers, and are useful models for periodic phases like in electrical engineering and quantum mechanics.
The techniques are also often useful for [evaluating integrals](https://en.wikipedia.org/wiki/Contour_integration), solving [two-dimensional electrostatics](https://en.wikipedia.org/wiki/Harmonic_function#Connections_with_complex_function_theory) and [decomposing periodic signals](https://en.wikipedia.org/wiki/Fourier_analysis).

Most of mathematical analysis, topology and measure theory is about [inapplicable abtruse examples](/sunk-cost-pure-maths).
This is to the extent that there's a whole book on [Counterexamples in Topology](https://en.wikipedia.org/wiki/Counterexamples_in_Topology).
Curiously in complex analysis all these strange examples like nowhere-differentiable functions are inadmissible, and leave well-behaved functions with useful techniques to handle them.

Complex numbers start with adding solutions i and -i to the equation $$ x^2 = -1 $$.
It turns out that when you extend real (or algebraic) numbers with this then it is enough to *factorise* any polynomial into linear factors.
This is called the [Fundamental Theorem of Algebra](https://en.wikipedia.org/wiki/Fundamental_theorem_of_algebra).
Understanding that a real polynomial is invariant under complex conjugation (that is interchanging i and -i), the non-real solutions must come in conjugate pairs.
Consequently the real factors are linear, or complex quadratic when they are a conjugate complex pair.

A function $$ f(z) = f(x+iy) $$ is holomorphic (complex differentiable) when $$ \frac{df}{d\overline{z}} = \frac{1}{2} \left( \frac{df}{dx} - \frac{1}{i} \frac{df}{dy} \right ) $$ is zero, and the derivative is $$ \frac{df}{d\bar{z}} = \frac{1}{2} \left( \frac{df}{dx} + \frac{1}{i} \frac{df}{dy} \right ) $$.
This comes straight from evaluating an epsilon-delta definition along real and imaginary lines.
It follows immediately that a holomorphic function is harmonic as a real valued function of two variables $$ \frac{d^2f}{dzd\overline{z}} = \frac{d^2f}{dx^2} + \frac{d^2f}{dy^2} = 0 $$.

A holomorphic function has a lot of the same [regularity properties](https://en.wikipedia.org/wiki/Harmonic_function#Properties_of_harmonic_functions) as harmonic functions.
The maximum of the absolute value occurs on the boundary of a set, not in the interior.
Moreover any holomorphic function is analytic: that is it can be locally expanded in a power series about a point $$ f(z) = \sum_{n=0}^{\infty} a_n (z - c)^n $$.
So having one complex derivative means it's infinitely differentiable and the derivatives at one point govern behaviour on the whole domain, in contrast to counterexamples in real analysis (and there are [even weaker conditions](https://en.wikipedia.org/wiki/Morera's_theorem)).
Even with real analytic functions like $$ \frac{1}{1 + x^2} $$ complex analysis can shed light on why the radius of convergence of the power series about 0 is 1 (because the complex extension diverges at i and -i).

Extending to ratios of holomorphic functions gives [meromorphic functions](https://en.wikipedia.org/wiki/Meromorphic_function) which have [Laurent Series](https://en.wikipedia.org/wiki/Laurent_series) $$ f(z) = \sum_{n=-\infty}^{\infty} a_n (z - c)^n $$.
In fact the Cauchy Integral Formula relates the Laurent series at a point with the values around any curve $$ \gamma $$ surrounding that point via the contour integral:

$$ a_n=\frac{1}{2\pi i} \oint_{\gamma}\,\frac{f(z)}{(z-c)^{n+1}}\,dz $$

This can be used to ["analytically continue"](https://en.wikipedia.org/wiki/Analytic_continuation) a function by expanding the series beyond its radius of convergence.
These lead to functions useful for applications like the [Gamma Function](https://en.wikipedia.org/wiki/Gamma_function) which is a generalisation of factorial to complex numbers, the related [Beta Function](/beta-function) useful in statistics, and the [Riemann zeta function](https://en.wikipedia.org/wiki/Riemann_zeta_function) which has applications in [Quantum Field fluctuations between two plates in a vacuum](https://en.wikipedia.org/wiki/Casimir_effect#Derivation_of_Casimir_effect_assuming_zeta-regularization).

A lot of these ideas extend to higher complex dimensions, however I'm not sure about their utility.
If a function is holomorphic with respect to each variable separately, it is [holomorphic with respect to all of them](https://en.wikipedia.org/wiki/Hartogs%27s_theorem).
In particular variations of meromorphic functions and the Cauchy Integral Formula apply.

Finally a brief word on *biholomorphisms*, holomorphic functions with a holomorphic inverse, which relate to the structure of complex domains.
The Riemann Sphere is obtained by adding a point at infinity to the complex plane. 
The biholomorphisms are the [Möbius transformation](https://en.wikipedia.org/wiki/M%C3%B6bius_transformation) which can map any 3 points of the Riemann sphere to any other 3 points.
In particular this implies that the biholomorphisms of the complex plane are just linear functions (since they are Möbius transformations that leave the point at infinity invariant); compare this with all the possible continuously differentiable bijections of the real plane.

The [Riemann Mapping Theorem](https://en.wikipedia.org/wiki/Riemann_mapping_theorem) says that all non-empty, open, simply connected (i.e. no "holes") subset of the plane have biholomorphisms, so they are equivalent from a complex perspective.
In particular the unit ball has the two-dimensional family of biholomorphisms $$ T_a(z) = \frac{z - a}{1 - \overline{a} z} e^{2 \pi i \theta} $$.
When there are holes there is additional structure; for example an [annulus](https://en.wikipedia.org/wiki/Annulus_(mathematics)#Complex_structure) has a single holomorphic invariant related to the ratio of the outer to inner radius.

This nice automorphism structure breaks down in higher dimensions; in particular I believe the automorphisms of the 2-complex dimensional space are unknown.
The "niceness" of the structure of the complex plane is likely related to its algebraic structure.
I don't know of many uses of these structural ideas, except in understanding 2-dimensional harmonic functions (e.g. in 2d fluid mechanics and electrostatics).

Complex analysis is, strangely, more practical than real analysis.
The tools like laurent Series, contour integrals and the Gamma Function pop up in surprisingly many applications.