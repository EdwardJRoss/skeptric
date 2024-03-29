---
date: 2011-05-21 11:06:48+00:00
image: /images/harmonic_oscillator.png
title: Symmetry, Lie Algebras and Differential Equations Part 1
---

There is a deep relationship between being able to solve a differential equation and its symmetries. Much of the theory of second order linear differential equations is really the theory of infinite dimensional linear algebra. In particular Sturm-Liouville theory is the diagonalization of an infinite dimensional Hermitian operator. However there are deeper relationships, as Miller points out in “Lie theory and special functions”; the relationships between special functions such as Rodrigues’ formulae are related to the Lie algebra and symmetries of the system. Even better in some cases the solutions can be found almost entirely algebraically. Some examples from physics come from the Simple Harmonic Oscillator, the theory of Angular Momentum and the Kepler Problem (using the Laplace Runge Lenz vector). The rest of this article will be devoted to exploring a special case of these relations the Quantum Simple Harmonic Oscillator.


<!--more-->


We begin with trying to solve the differential equation  $-\frac{1}{2m} f''(x) + \frac{k}{2} x^2 f(x) = \lambda f(x)$  for some real positive constants  $m$ ,  $k$  and $latex\lambda$ with the boundary conditions f vanishes at infinity. This is an eigenvalue equation; this can’t be solved for any constants but only for particular values of  $\lambda$  for a fixed k and m. By dilations (that is, rescaling units) we can assume without loss of generality  $m=1$  and  $k=1$ . It is useful to define the **momentum operator**  $p=-i \frac{\mathrm{d}}{\mathrm{d}x}$  – this makes everything more physics-like. If this isn’t familiar to you just substitute  $-i \frac{\mathrm{d}}{\mathrm{d}x}$  wherever you see a p.


(The i is chosen to make the operator Hermitian with respect to the  $L^2$  inner product:  $\int_{-\infty}^{\infty}{f(x)}^* p g(x) = \int_{-\infty}^{\infty} \left(pf(x)\right)^* g(x)$ ; where * denotes complex conjugation and f and g are zero at infinity. This identity follows immediately from integration by parts).


Introducing the Hamiltonian operator  $H=\frac{1}{2}\left(p^2+x^2\right)$  the differential equation is then the eigenvalue equation  $H f(x) = \lambda f(x)$  (a form familiar to physicists). The Hamiltonian operator has an obvious symmetry to it: it is invariant under rotations in x-p space. That is it is invariant under transformations of the form:


 $\begin{bmatrix} x' \\ p' \end{bmatrix} = \begin{bmatrix} \cos \theta & -\sin \theta \\ \sin \theta & \cos \theta \end{bmatrix} \begin{bmatrix} x \\ p \end{bmatrix}$ .


Following the ideas of Sophus Lie we look at the infinitesimal transformations generating this, by taking the derivative at the identity  $\theta=0$  this gives  $x \to -p$ ,  $p \to x$ ; in x-p space it is given by the matrix  $\begin{bmatrix} 0 & -1 \\ 1 & 0\end{bmatrix}$


This transformation is precisely the Fourier transform:  $f(x) \to \widehat{f}(k)=\frac{1}{\sqrt{2\pi}}\int_{-\infty}^{\infty} f(x) e^{-i k x}$ . In particular integration by parts and differentiating under the integral respectively it follows  $\widehat{xf(x)}=i\widehat{f}'(k)$  and  $\widehat{-if'(x)} = k \widehat{f}(k)$ , so as an operator on functions  $x \to -p$  and  $p \to x$ .


Now since the square of the Fourier transform in x-p space is negative the identity it has eigenvalues -i and +i and corresponding eigenvectors  $a = \frac{1}{\sqrt{2}}(x+ip)$  and  $a^{\dagger} = \frac{1}{\sqrt{2}} (x- ip)$ .


Now we introduce the **commutator** of operators  $[A,B]=AB-BA=-[B,A]$  and in particular  $[x,p]=i \mbox{Id}$  (since x and its derivative don’t commute). Consequently by linearity  $[a,a^{\dagger}]=1$ .


Simple calculations show that  $a^{\dagger}a = H -1/2$ ,  $a a^{\dagger} = H + 1/2$ ,  $[H,a]=-a$  and  $[H,a^{\dagger}]=a^{\dagger}$ . These last two relations allow us to find the spectrum of H, that is the values of  $\lambda$  for which the differential equation is solvable!


If the differential equation can be solved for some  $\lambda$ ,  $H f = \lambda f$  then using the commutation relations shows  $H(a f) = (\lambda-1) (af)$  and  $H(a^{\dagger}f) = (\lambda +1) (a^{\dagger}f)$ . Thus  $a$  lowers the eigenvalue by 1 and is called a **lowering operator**, and  $a^{\dagger}$  raises the eigenvalue by 1 and is called a **raising operator.**


However we can not lower indefinitely: H is positive semidefinite,  $H=\frac{1}{2} (x^{\dagger}x + p^{\dagger}p)$  (where the dagger indicates Hermitian conjugation with respect to the  $L^2$  inner product), so $\lambda$ must be non-negative. Thus there is a function  $f_0(x)$  for which  $a f_0(x) = 0$  (which of course satisfies the differential equation trivially). On this state  $H f_0(x) = (a^{\dagger} a + 1/2) f_0(x) = 1/2 f_0(x)$ .


Moreover since any arbitrary solution can be brought to  $f_0$  by repeated lowering (applications of a), and lowering then raising gives a multiple of the original function every solution can be obtained by raising  $f_0$ . Thus the only possible eigenvalues are n+1/2 for n=0,1,2,….


What are the corresponding eigenvectors? Well  $a f_0 = 0$  implies that  $x f_0(x) + f_0'(x) =0$ , which has solutions  $f_0(x) = A e^{-\frac{x^2}{2}}$  for some constant  $A$ . Then the solution with  $\lambda = n + 1/2$  is up to a constant factor  $(a^{\dagger})^n f_0(x) = A_n \left(x - \frac{\mathrm{d}}{\mathrm{d}x}\right)^n e^{-\frac{x^2}{2}} = H_n(x) e^{-\frac{x^2}{2}}$  where  $H_n(x)$  are the **Hermite polynomials**. Consequently we have found all solutions of the second order differential equation just by solving a first order differential equation! (They can also easily be normalized algebraically; that is without doing any integrals, but I won’t show that here).


It is interesting to note all these solutions are invariant under Fourier transform. This is of course a consequence of the Hamiltonian being invariant under Fourier transform, F; if  $Hf = \lambda f$  then  $F H f = (FHF^{-1}F)f = HFf$  and thus  $\lambda F f = H (Ff)$ .


From an abstract point of view what have we done? We have taken an algebra of operators on some Hilbert space generated by self-adjoint operators  $x$  and  $p$  satisfying  $xp-px=i \text{id}$  (notice that this implies the vector space can’t be finite dimensional; take the trace of each side). Using this we have shown that the positive definite Hermitian operator  $H = \frac{1}{2} (x^2 + p^2)$  has eigenvalues n + 1/2 for n=0,1,2,….


We could choose an explicit representation: the Hilbert space is the space of square integrable functions, x is the multiplication operator and  $p = -i \frac{\mathrm{d}}{\mathrm{d}x}$ , then in this basis the eigenequation is the differential equation we started with. The solutions in this basis are the Hermite polynomials multiplied by a Gaussian; notice that these functions are orthogonal and complete in  $L^2$  being all the eigenfunctions of a Hermitian operator. The formula for the eigenfunctions in terms of raising operators gives rise to a Rodrigues formula for the Hermite polynomials.


However there is nothing canonical about this choice of representation, a different representation is given by the Fourier transform, which acts as a change of basis. That the Hamiltonian is invariant under the Fourier transform means  $FHF^{-1}=H$  or  $[F,H]=0$ .


The nicest choice of basis is the one in which H is the (countably infinite dimensional) diagonal matrix with entries 1/2,3/2,5/2,…. It is easy to see that a is the matrix with 1s one row below the diagonal and zeros everywhere else  $a=\begin{bmatrix} 0 & 0 & 0 & \ldots \\ 1 & 0 & 0 & \ldots \\ 0 & 1 & 0 & \ldots \\ &&& \ddots \end{bmatrix}$  and  $a^{\dagger}$  is its transpose. Representations for x and p can be obtained from  $x = \frac{1}{\sqrt{2}}(a + a^{\dagger})$  and  $p = \frac{1}{2i} (a - a^{\dagger})$ .


It is worth noting that in this derivation it wasn’t enough to have a Lie algebra, that is a Lie bracket, we also needed a multiplication over which the Lie bracket is the commutator – that is a representation.