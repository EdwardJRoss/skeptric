---
date: 2012-08-13 12:34:04+00:00
image: /images/tangent_vector.svg
title: Geometry and topology of division rings
---

Following from my [last post](/geometry-of-division-rings/) (and Veblen and Young’s Projective Geometry) consider a projective plane satisfying the axioms:




1.  Given two distinct points there is a unique line that both points lie on
1.  Each line has at least three points which lie on it
1.  Given a triangle any line that intersects two sides of the triangle intersects the third.
1.  All points are spanned by d+1 points and no fewer.



Then for d>=3 is equivalent to the projective space of lines over a division ring (or skew field).


Kolmogorov asked the question what projective spaces can we do analysis on? In order to do things such as find tangent lines we are going to need some sort of topology.


<!--more-->


Kolmogorov apparently proved that for a (Desarguian) projective space if the set of points is compact and infinite, the set of lines is compact and the function mapping two distinct points to the line they lie on is continuous then the underlying division ring is infinite and locally compact (in a paper translated as “The Axiomatics of Projective Geometry” in Selected works of A. N . Kolmogorov edited by V. M. Tikhomirov). Such an object is called a continuous projective geometry.


In response Pontryagin proved (see his book “Topological Groups”) proved that every locally compact infinite division ring contains one of: the real numbers, the [p-adic numbers](http://en.wikipedia.org/wiki/P-adic_numbers), the power series over the integers modulo p (p prime). Moreover we can classify these by their connectedness and characteristic: if the division ring is connected it contains the real numbers, otherwise it is totally disconnected.


Combining this with the [Frobenius theorem](http://en.wikipedia.org/wiki/Frobenius_theorem_%28real_division_algebras%29) we have the following: A locally compact connected field is isomorphic to the real numbers, the complex numbers or the quaternions.


Separation theorems allow us to define regions and boundaries of regions, so we can start to talk about ‘relative lengths’ and ‘relative areas’. One way to approach the separation theorems in projective geometry is via ordered fields: [Veblen and Young](http://archive.org/details/projectivegeomet028875mbp) pursue such an approach; of course this doesn’t apply to an unordered field such as the complex numbers. Another is via topology; e.g. a line separates the plane it lies in into two (topologically) connected sets.


In some sense all this indicates the “natural” projective spaces to do calculus in are precisely the projective spaces over the real numbers, complex numbers or quaternions (and maybe the octonions?).


The calculus of real and complex numbers is well known; is there a corresponding exterior differential calculus of quaternions? Given two n-simplices in an n-dimensional affine space, there is a unique affine transformation from one to the other. The ratio of their hypervolumes is the determinant of the linear transformation. Is there an analogous determinant for quaternions (or octonions)?


Essentially [no](http://www.math.nus.edu.sg/aslaksen/papers/S-QD.pdf); Dieudonne [extended](http://www.numdam.org/item?id=BSMF_1943__71__27_0) the determinant to a non-commutative field by defining it as a map from matrices to the the division ring over its commutator subgroup (see Artin’s Geometric Algebra for details). This is about as good as you can do; any map from the general linear group on an n-dimensional (right) quaternion vector space to the quaternions that satisfies




1.  det(AB) = det(A) det(B) (multiplicative)
1.  det(A) = 0 iff A is not invertible (homomorphism)
1.  If E is a matrix with 1s along the diagonal and exactly one other non-zero entry then det(E) = 1 (invariance under skew transforms)



then the image of the determinant is commutative.


To see an example of such an obstruction, consider the 2×2 quaternion matrices. Given a diagonal matrix  $$\left(\begin{array}{cc} a & 0 \\ 0 & b \end{array}\right)$$  would the determinant be ab or ba? For a commutative ring, a 2×2 matrix satisfies  $$A^2 - \mathrm{tr}(A) A + \mathrm{det}(A)I = 0$$ . A little experimentation shows there isn’t a similar formula for the quaternions (we can’t get rid of the off-diagonal elements). In fact taking the trace gives the formula for the determinant  $$\mathrm{det}(A) = \frac{\mathrm{tr}(A)^2 - \mathrm{tr}(A^2)}{2}$$ . If we try to apply this to a quaternion matrix we get  $$\mathrm{det} \left(\begin{array}{cc} a & b \\ c & d \end{array}\right) = ad + da - bc -cb$$ . Notice that since ij=-ji, ik=-ki, jk=-kj this yields a real number. (The parallel actually extends into the spectral theory of quatenionic matrices)


In [fact](http://arXiv.org/abs/math-ph/9907015v2) given any two distinct maps  $$\mathrm{det} \colon \mathbb{H} \to \mathbb{R}_{\geq 0}$$  satisfying axioms 1-3, one is a real power of the other. One way to construct such a determinant is to notice that quaternions can be represented by 2×2 complex matrices of the form  $$\left(\begin{array}{cc} a & b \\ -\bar{b} & a \end{array}\right)$$  where a and b are complex numbers. We can then take the absolute value of the complex determinant (this is called the Study determinant, which is the square of the Dieudonne determinant). Alternatively we could repeat a similar expansion for complex numbers in terms of real numbers, giving a quaternion as a 4×4 real matrix. We then define the determinant of an nxn quaternion matrix, as the determinant of the corresponding 4nx4n real matrix; this is called the q-determinant and is the square of the Study determinant.