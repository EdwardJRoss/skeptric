---
date: 2011-05-21 17:03:57+00:00
image: /images/kepler_areas.svg
title: Symmetry, Lie Algebras and Differential Equations Part 3
---

There is a deep relationship between the technique of **separation of variables** for solving partial differential equations and the symmetries of the underlying differential equations, as well as the special functions that often arise in this procedure.


<!--more-->


As a motivating example consider the three-dimensional quantum Kepler problem  $H \psi = E \psi$  for  $H=p^2 + \frac{1}{\sqrt{r^2}}$ , where r is the 3 dimensional position operators and p is the 3 dimensional momentum vector in the previous article (that is the components of r and p satisfy the algebraic relation  $x_ip_j-p_jx_i = i \delta_{ij} \mbox{Id}$ ).


The Kepler problem clearly has a rotational symmetry about any axis, and so the angular momentum is conserved and the operators  $H, L_z, L^2$  form a mutually commuting set. In fact there is another conserved quantity, unique to the 1/r potential, the **Laplace-Runge-Lenz vector**,  $A = \frac{1}{2} (p \times L - L \times p) - \hat{r}$  where  $\hat{r}=\frac{r}{\sqrt{r^2}}$  is the unit vector – in classical mechanics the conservation of this vector corresponds to the fact orbits are conic sections. As it turns out angular momentum ladder operators can be built from  $L_z$  and  $A_z$  for the negative eigenspace of  $H$  and can be used to find the spectrum and eigenfunctions as we did for angular momentum and the simple harmonic oscillator; for the details see e.g. Thaller’s Advanced Visual Quantum Mechanics.


The corresponding symmetry is quite subtle: in momentum space the Kepler problem can be viewed via stereographic projection as a [free particle on a three sphere](http://rmp.aps.org/abstract/RMP/v38/i2/p330_1). With respect to  $H,L^2,L_z$  this symmetry is related to the fact the eigenvalues of  $H$  are independent of those of  $L_z$ .


It is interesting that according to [Cordani’s monograph on the Kepler Problem](http://books.google.com.au/books?id=RiQJQjOwU3sC&pg=PA444&lpg=PA444&dq=cordani+kepler+problem&source=bl&ots=bidhP5rtWy&sig=ygPfbbbFgyhJtykBL2_n6T4Vskw&hl=en&ei=K73XTcC3IoXAsAPjmLS5Bw&sa=X&oi=book_result&ct=result&resnum=1&ved=0CBgQ6AEwAA#v=onepage&q&f=false) the Kepler problem is separable in exactly 4 coordinate systems, and the corresponding “first integrals” (separation constants) are given by combinations of the operators above.




1.  Spherical Coordinates:  $H,L_z,L^2$
1.  Parabolic Coordinates:  $H,A_z,L_z$
1.  Confocal Elliptic Coordinates:  $H,L^2-a A_z,L_z$
1.  Conical Coordinates:  $H,L^2, L_x^2 - a^2(L_x^2+L_y^2)$



the eigenfunctions in these coordinates are some sort of hypergeometric functions. From a physical perspective it is also interesting that these mean the Kepler problem remains separable under certain perturbations (but the spectrum can no longer be derived algebraically): spherical coordinates are invariant under spherical perturbations and if the particle is charged interaction with a constant magnetic field; parabolic coordinates are invariant under perturbation by a constant force along the z-axis, and if the particle is charged interaction with a constant electric field; confocal elliptic coordinates are invariant under perturbation by another central force at the other focus, this corresponds to Euler’s restricted 3-body problem; I’m not sure about conical coordinates.


In fact as shown in Morse and Feschbach – Methods of Theoretical Physics the Helmholtz/Diffusion equation  $\nabla^2 \psi = \lambda \psi$  (which for  $\lambda < 0$  is the Schrodinger equation for a free particle) in 3 dimensions is separable in **11 coordinate systems**. There is some relationship between the separability, symmetry, conserved quantities and special functions, I will outline some special cases:




1.  Cartesian coordinates. Translational symmetry.  $p_x,p_y,p_z$ . Coordinate functions  $x,y,z$ .
1.  Circular cylindrical coordinates. 1D translational + 1D rotational .  $H=p^2,p_z,L_z$ . Bessel functions.
1.  Spherical coordinates. Rotational symmetry.  $H=p^2,L_z,L^2$ . Spherical harmonics.



I will leave open some questions I would like to answer:


What is this relationship between the integrability of a dynamical system, its symmetries, representations of its Lie algebra and the special functions?


Are there other physically important systems that are separable – breaking some but not all of the symmetries?


There are many hints to these questions in the literature; for example [Miller’s expositions ](http://www.ima.umn.edu/%7Emiller/separationofvariables.html)on the relation between special function and symmetry. There are derivations of possible coordinate systems an equation separates in using something called a Stäckel transformation (this is what both Morse and Feschbach and Cordani use). Lie algebras were in fact invented to understand the solubility of differential equations in the same way Galois groups use symmetry to understand the solubility of algebraic equations (but unfortunately algebraists removed it from its roots).


As to the second question there is Bruns’ theorem: the only conserved quantities in the classical three body problem are the centre of mass, momentum, angular momentum and energy. So in non-relativistic quantum mechanics we would expect no extra conserved quantities for the three body problem (and consequently the many body problem) – any separable systems must be effectively one or two particle (like Euler’s restricted 3-body problem).