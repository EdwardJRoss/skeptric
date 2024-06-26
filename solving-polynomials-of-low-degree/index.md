---
date: 2011-01-03 02:45:58+00:00
image: /images/cubic_eqn.jpg
title: Solving polynomials of degree 2,3 and 4
---

$$\newcommand\nth{n^{\mathrm{th}}}$$

It is well known in mathematics that it is possible to find the roots of a general quadratic, cubic or quartic in terms of radicals (linear combinations and products of $\nth$ roots). Another way of saying this is that the equation
 $a x^4+b x^3+c x^2 + d x + e = 0$  can be solved for any complex constants $a$,$b$,$c$,$d$, and $e$ if one can solve the equation  $x^n-t=0$  for $n \in \{2,3\}$ ($1$ being trivial) ($t$ may be an algebraic combination of solutions of  $x^n-s$  for a variety of $s$ which are algebraic combinations of $a$,$b$,$c$,$d$ and $e$). This is not true for the quintic.


<!--more-->


As an aside radicals are often thought of as somehow “simple” or “special”, but really this is only as true as special functions are special – just because we are used to them and can calculate them easily we consider them special. The solutions to the equation  $x^5+x-a=0$  aren’t in a general sense less fundamental than the solutions to the equation  $x^5-a=0$  so we should be careful in ascribing too much meaning to the insolubility of the general quintic by radicals.


However in principle these solutions by radicals give us really efficient ways of finding all the roots of polynomials of degrees  $\leq 4$  to an arbitrary accuracy; just accurately calculate the $\nth$ roots and apply some algebra (being mindful of roundoff error).


This is very well known in the case of the quadratic; the solutions to


 $a x^2 + b x + c = 0$  ($a \neq 0$) are given by the quadratic formula  $x=\frac{-b + (b^2-4 a c)^{1/2}}{2 a}$  (since any non-zero number has two square roots, this gives two solutions except when  $b^2=4 a c$ ).


This is also very simple to understand geometrically and algebraically. Let’s rescale our polynomial (which doesn’t change the zeros) giving


 $x^2 - 2 d x - e = 0$  ($d=-b/2a$, $e=-c/a$). Algebraically the trick is completing the square to eliminate the linear term:


 $(x-d)^2 - e = d^2$ . Geometrically the replacement $x-d$ for $x$ is just a change in origin: we change the origin to the extremum (maximum or minimum) of the quadratic about which the quadratic is symmetric – both roots are equidistant from the extremum.
We then rearrange algebraically to find this distance:


 $x-d = (d^2+e)^{1/2}$  giving the quadratic formula.


(A technical detail: The function  $f(x)=x^n$  mapping  $[0,\infty)$  to itself for any positive integer $n$ is a bijection, so there is a unique inverse  $f^{-1}(x)=\sqrt[n]{x}$ . Let  $w=r e^{i \theta}$  be any complex number, then for $w$ not zero there are precisely $n$ solutions in the complex plane to  $z^n = w$ , given by  $z = \sqrt[n]{r} e^{i (\theta+2 k \pi)/n}$  for $k$ an integer (since  $e^{2 i \pi} = 1$  only $n$ of these are distinct). When I write  $w^{\frac{1}/{n}}$  I am referring to these $n$ solutions.)


When people write the quadratic formula they often write  $\pm\sqrt{\ldots}$ , which is fine for the real case, but in the complex case it is less obvious what is meant by square root (you could take the principal value: writing  $\theta$  above in  $(-\pi,\pi]$  and referring to the $\nth$ root as $k=0$ but this is somewhat arbitrary, and masks the symmetry of the solutions).


The solution to the cubic can similarly be written out: see e.g. [here](http://www.math.vanderbilt.edu/~schectex/courses/cubic/) – it is much more involved than the quadratic formula. (Incidentally the history of the cubic formula is somewhat obscured: see the [Math World](http://mathworld.wolfram.com/CubicFormula.html) article). Written out in full the cubic formula is quite hard to interpret, but a medical professional has written a [beautiful paper](http://www.nickalls.org/dick/papers/maths/cubic1993.pdf) explaining the solution in a geometrically illuminating manner.


An outline of the solution is first “completing the cube” to remove the $x^2$ term (that is shifting the origin such that the sum of the roots is zero), giving an equation like  $x^3 + a x + b =0$ . Then the essential algebraic trick is noting  $(p+q)^3 - 3 p q (p+q) - (p^3+q^3)$  and setting $x=p+q$, and solving using the equations $3 p q = – a$ and  $p^3+q^3=-b$ . This gives identical quadratics in $p^3$ and $q^3$: one solution is $q^3$ and the other is $p^3$ (since there is complete symmetry in $p$ and $q$ it doesn’t matter which). Then all that remains is to take the $\frac{1}{3}$ power of say $p$ (which has $3$ solutions) and finding $q$ using $3pq = -a$. This gives all three solutions. [One has to be very careful of the explicit formula in terms of the coefficients in the complex case: it involves the sum of two cube roots, and there are 3 possible solutions to each cube root giving a total of 9 solutions, but 6 of these are spurious: we need to enforce the restriction of their product).


The cubic formula, however, is rather nasty in that it may make the solutions look more complicated than they are. For instance consider the cubic


 $(x-1)(x^2+1)=x^3-x^2+x-1=0$ , the solutions given by this algorithm are


 $\frac{1+a+b}{3}$  and  $\pm i \frac{a-b}{2 \sqrt{3}}$  where


 $a=\sqrt[3]{10+6 \sqrt{3}}$  and  $b=\sqrt[3]{10-6 \sqrt{3}}$ .


It is not obvious at all that these solutions correspond to $1$, $i$, and $-i$ unless one realises  $(1 \pm \sqrt{3})^3=10 \pm 6 \sqrt{3}$  and so  $a=1+\sqrt{3}$  and  $b = 1 - \sqrt{3}$ .


Even if all the coefficients are real, if there are 3 real roots we necessarily go through the complex numbers using this procedure. For example


 $(x-1)x(x+1)=x^3-x$  gives pairs $p$,$q$ (or equivalently $q$,$p$)  $(\frac{1}{2}+\frac{i}{2\sqrt{3}},\frac{1}{2}-\frac{i}{2\sqrt{3}}), (-\frac{1}{2}+\frac{i}{2\sqrt{3}},-\frac{1}{2}-\frac{i}{2\sqrt{3}}), (\frac{i}{\sqrt{3}},-\frac{i}{\sqrt{3}})$ . (In fact this is probably one of the reasons the complex numbers were accepted as a valid entity – they arose in solving real cubics with real roots).


These things aside the cubic formula is still a very fast way to find the roots of a cubic numerically.


Solving quartics is not much more tricky than solving cubics (though in fact a necessary step in solving a quartic is solving a cubic). Good explanations can be found [here](http://www.sosmath.com/algebra/factor/fac12/fac12.html) and a more symmetric explanation [here](http://www.nickalls.org/dick/papers/maths/quartic2009.pdf).


So we can solve quadratics, cubics and quartics by taking square and cube roots. How useful is this? Is it faster and more accurate than general techniques for finding the roots of polynomials?


What is the underlying pattern in these solutions? What polynomials of higher degree can be solved by radicals? What *ultraradicals* (solutions of equations such as  $x^5 + x - a = 0$ ) are needed to solve higher order polynomials? [I believe the answers to this question lie in Galois theory].


Would it be more efficient to compute the roots of a quintic by hyperradicals or by a more general procedure?


These questions do have import in daily mathematics. To calculate an integral by Gaussian quadrature the zeros of special classes of polynomials is required (in fact in this case there are a series of polynomials of degree $0,1,2,3,\ldots$ with all their roots real such that the zeros interlace – i.e. between every two zeros of the degree $n$ polynomial there is a zero of the degree $n+1$ polynomial).


In fact often in applied mathematics functions are approximated by polynomials, or known functions multiplied by polynomials, the zeros of which may have some import.