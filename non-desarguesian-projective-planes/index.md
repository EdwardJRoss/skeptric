---
date: 2012-09-22 11:43:02+00:00
image: /images/desargues_theorem.svg
title: Non-desarguesian projective planes
---

There are two main constructions of a projective space.


<!--more-->


Suppose we have an n-dimensional affine space over a skewfield k. We can construct an (n-1)-dimensional projective space by taking the pencils of lines as points (a *pencil* of lines is a complete set of parallel lines; that is the lines under the equivalence relation of parallelism), and 2-pencils of planes (complete sets of parallel planes) as lines, and so on. A less invariant way is to choose an origin, so we get a vector space; then each pencil is represented by a unique line through the origin, and each 2-pencil is represented by a unique plane through the origin; thus the points of projective space are lines through the origin and the lines are planes through the origin. Thus a projective space of dimension n can be constructed as the quotient of an affine or vector space of dimension n+1.


Now given an n-dimesional projective space, consider an arbitrary (n-1)-dimensional projective subspace P. We can use this to form an affine subspace of dimension n: The points are the projective points not on P, the lines are the set of all lines not in P, and two lines are said to be parallel if they intersect at the same point of P. Given any line l and a point L not on that line the line through L and (the intersection of l with P) is the only line parallel to l through L. Thus a projective space of dimension n can be decomposed into an affine space of dimension n and a projective space of dimension (n-1).


This second construction can in fact be extended to all projective planes. Given a skewfield we have a natural way of representing two dimensional (left)-affine lines, as the set of points (x,y) satisfying the equation y = xa+b for some a and b. For general planes we extend the notion of this triple T(x,a,b)=xa+b. A **planar ternary ring** is a set R with at least two elements and a ternary operation  $$T: R \times R \times R \to R$$  that satisfies:




*  For each a, b, c in R there exists a unique x in R such that T(a,b,x) = c
*  For each  $$a \neq a'$$ , b, b’ in R there exists a unique x in R such that T(x, a, b) = T(x, a’, b’)
*  For each  $$a \neq a'$$ , b, b’ in R there exists a unique pair (x, y) in  $$R \times R$$  such that T(a, x, y) = b and T(a’, x, y) = b’.



Then the following construction, roughly following [Weibel](http://www.ams.org/notices/200710/tx071001294p.pdf)(who’s following [Hall](http://www.ams.org/journals/tran/1943-054-02/S0002-9947-1943-0008892-4/S0002-9947-1943-0008892-4.pdf)), yields a projective plane by approximately the reverse of the construction above. The points are  $$R \times R$$  (the ordinary points), R and  $$\infty$$  (the projective line at infinity). The lines are  $$\{(x,y) \in R \times R | y = T(x, a, b)\} \cup \{a\}$$  for each a in R and b in R,  $$\{(c,y) | y \in R\} \cup \{\infty\}$$  for each c in R (these are the ordinary lines) and  $$R \cup \{\infty\}$$  (the line at infinity). One can check that this does in fact define a projective plane.


Two ternary rings (R, T) and (R, T’) are *comparative *if there exist permutations on R  $$\alpha, \beta, \{\phi_x\}_{x \in R}, \{\psi_x\}_{x \in R}$$  such that  $$\phi_x \circ T(x, a, b) = T'( \alpha \circ x, \beta \circ a, \psi_a \circ b)$$  for all a,b, x in R. It is easy to show comparative rings yield isomorphic projective planes.


We can conversely construct a ternary ring from a projective plane, but there are clearly a lot of choices to be made; we need to choose a line at infinity, a y-axis and some isomorphisms between the y-axis minus it’s intersection with infinity and other lines, and use the lines and these isomorphisms to define T(x, a, b). Different choices need not yield comparative ternary rings (see [here](http://www.springerlink.com/content/4fmhjmhy8eqxenrw/?MUD=MP) for a necessary and sufficient condition).


If we add additional structure this is sometimes unique; for instance given a projective plane coordinitised by an **alternative division ring**, isomorphic projective planes yield isomorphic alternative division rings (see Bruck and Kleinfeld – The structure of alternative division rings, for a proof). I’m not sure if this is the best result one can obtain; in the finite case the coordinate rings are isomorphic (in the sense of Hall ternary rings; ternary rings with a 1 and a 0) if and only if they are a finite field.


It’s interesting to note a generalised construction of projective planes of the first type for every division algebra over the real numbers (including the octonions). The points and lines of the space are each the 3-dimensional Hermitian idempotents of unit trace (that is 3×3 matrices P satisfying  $$P = P^\dagger$$ ,  $$P^2=P$$  so P is a projection, and trace(P)=1). A point P lies on a line Q when PQ+QP=0. It can be shown this is a projective plane (see Conway and Smith, On Quaternions and Octionions, for the details).