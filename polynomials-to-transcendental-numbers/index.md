---
date: 2011-06-08 14:10:28+00:00
image: /images/pi.jpg
title: From polynomials to transcendental numbers
---

In a [previous post](/solving-polynomials-of-low-degree) I discussed finding the zeros of low degree polynomials; I want to extend that discussion to algorithmically finding the zeros of polynomials, more on solving the quintic and a brief discussion of transcendental numbers.


<!--more-->


As we were taught in high school the roots of the quadratic equation  $a x^2 + b x + c=0$  can be found by completing the square, giving  $x = \frac{-b \pm (b^2-4ac)^{1/2}}{2a}$ .


There are some problems with implementing it: firstly we need to be able to take square roots (that is solve  $x^2=a$ ). This isn’t too bad there are [lots of algorithms](http://en.wikipedia.org/wiki/Methods_of_computing_square_roots), geometrically we can do it with a compass and a straightedge (indeed this is where the first examples of irrational numbers came from) and recently it’s been done using [DNA](http://www.nature.com/news/2011/110602/full/news.2011.343.html). A more serious problem is round-off error: if  $b^2 \gg ac$  then  $|b| \approx |\sqrt{b^2-4ac}|$  and so if you only calculate this using a few decimal places there will be significant roundoff error in  $b - \mbox{sgn}(b) \sqrt{b^2-4ac}$  (whether this is important depends on the application). A simple workaround is to notice that at most one of the roots will have this roundoff error and the product of the roots is c so we can use this to find the other one.


A much more detailed analysis of algorithms for solving the quadratic are in a [pair ](http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=1528437)of [articles](http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=1607926) by James Blinn. In fact he also has a [series](http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=1626190) of [articles](http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=1652931) on [solving](http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=4012570) the [cubic](http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=4052506) using the analagous [cubic formula](http://ieeexplore.ieee.org/xpl/freeabs_all.jsp?arnumber=4178164). One important thing to draw from this is the amount of work involved in recasting the classical quadratic and cubic formulae in a numerically stable way. Also as Blinn points out the utility of these methods depends heavily on your tools and application: depending on your computer (whether it be a pen and paper, an old fashioned calculator, a GPU or a molecular computer) it may be faster to iterate a solution than solve using a formula, and whether it makes a difference depends on how many polynomial equations you have to solve and how long each takes to solve. [ There are specific zero finding methods for polynomials, for example the [Jenkins-Traub algorithm](http://en.wikipedia.org/wiki/Jenkins-Traub_algorithm) that will converge much faster than generic methods such as Newton’s or gradient methods]. To solve a single cubic (or even a couple hundred) on a modern PC you wouldn’t notice a difference, but in some graphical applications you may need to solve thousands a second.


Incidentally Felix Klein found the quintic equation was tied up with the geometry of the icosahedron. In 1989 it was [shown](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.120.7132) further that the quintic could be solved by an **iterative algorithm**, which (I think) means that to each quintic is assigned a rational function and the roots can be found by repeatedly applying the rational function. The whole kit – from the insolubility by radicals to this algorithm – is explained in detail in this excellent set of [notes](http://people.reed.edu/~jerry/Quintic/quintic.html). (I would love an excuse to implement this algorithm in a stable manner).


Why limit ourselves to radicals? Why not consider  $\tan(\pi/15)$  a perfectly good solution (it’s in a familiar ‘nice’ form). This is the type of question [Timothy Chow asks](http://www-math.mit.edu/~tchow/closedform.pdf). More precisely he looks at the **‘EL numbers’**, the smallest subfield of the complex numbers closed under exponentiation and its compositional inverse, taking logarithms and asks what sorts of equations can you solve with it. The answer isn’t known, it lies in transcendental theory.


It’s possible, once you know [the trick](http://en.wikipedia.org/wiki/Transcendental_number#Sketch_of_a_proof_that_e_is_transcendental) to show  $e$  and  $\pi$  are transcendental. In fact the [Lindemann-Weierstrass theorem](http://en.wikipedia.org/wiki/Lindemann%E2%80%93Weierstrass_theorem) states that given linearly independent algebraic numbers over the rationals their exponentials are linearly independent over the rationals. The [Gelfond-Schneider theorem](http://en.wikipedia.org/wiki/Gelfond%E2%80%93Schneider_theorem) states that all values of  $\alpha^\beta$  are transcendental for  $\alpha \neq 0,1$  and  $\beta$  irrational. There are some [more theorems](http://en.wikipedia.org/wiki/Transcendence_theory) and a handful of other [transcendental numbers known](http://mathworld.wolfram.com/TranscendentalNumber.html) but a great deal is still unknown, for example are  $e + \pi$  and  $e \pi$  transcendental. The [constant problem](http://en.wikipedia.org/wiki/Constant_problem) of determining when a given transcendental function is zero (useful for computer algebra) has only been solved or proven algorithmically undecidable in certain cases.


A huge conjecture in transcendence theory is [Schnaul’s conjecture](http://en.wikipedia.org/wiki/Schanuel%27s_conjecture): given n complex numbers linearly independent over the rationals, then some collection of n terms taken from these numbers and their exponentials are algebraically independent. It would have strong implications: the Lindeman-Weirstrass and Gelfond-Schneider theorems are special cases, it would imply that Euler’s identity  $e^{i \pi} + 1 = 0$  is (in an appropriate sense) essentially the only algebraic relationship between  $\pi$  and  $e$ , and would bring us closer to understanding which algebraic and transcendental equations are solvable in the ‘EL numbers’ and their closure (the elementary numbers).


Of course one often talks as well about elementary functions (functions generated by constant functions, identity function and exponentiation under addition, multiplication, composition and their inverse operations) and it’s often said that  $\int_0^x e^{-t^2} \mathrm{d}x$  isn’t elementary. This apparently can be proved using Picard-Vessiot theory and differential galois theory. This is very closely related to my previous posts on integrable systems, Lie algebras and symmetries.


One last trick to leave you with. Solving one linear equation is easy, and in first year mathematics courses we are taught how to solve systems of linear equations. Since we’ve discussed solving one polynomial equation it’s natural to ask how would one solve a system of linear equations… one approach is a [Gröbner basis](http://math.berkeley.edu/~bernd/what-is.pdf)