---
date: 2013-10-27 12:02:36+00:00
image: /images/spectrum_iron.svg
title: Spectra of atoms
---

Why is a sodium lamp yellow? How can we determine the elemental composition of the sun? How does a Helium-neon laser can work?


To some degree all of these questions require knowing the spectra of atoms, which can in theory be calculated by Quantum mechanics. However the calculations of these spectra for arbitrary systems from first principles is prohibitively difficult and computationally intensive (which is why techniques such as Density Functional Theory are used).


This post will roughly outline to calculate the spectrum of the smaller atoms by explicitly diagonalising a matrix, whose elements are simple combinatorial quantities.


The non-relativistic Hamiltonian in the Born-Oppenheimer approximation of an n-electron atom in SI units is given by  $H = \sum_{i=1}^{n} \left( \frac{p_i^2}{2m} - \frac{Ze^2}{4 \pi \epsilon_0} \frac{1}{r_i} + \frac{e^2}{4 \pi \epsilon_0} \sum_{j > i} r_{ij} \right)$  where  $p_i$  is the momentum of the ith electron,  $r_i$  is its distance from the nucleus, $r_{ij}$ is the distance between the ith and jth electron, m is the mass of an electron, Z is the charge of the nucleus, and e is the charge of an electron.


To simplify matters choose units such that  $\frac{e^2}{4 \pi \epsilon_0} = 1$ ,  $m = \frac{1}{2}$  and  $\hbar = 1$ , these will be used in the rest of this article. Then  $H = \sum_{i=1}^{n} p_i^2 - \sum_{i=1}^{n} \frac{Z}{r_i} +\sum_{i=1}^{n}\sum_{j>i}^{n} \frac{1}{r_{ij}}$ . The terms correspond to the kinetic energy of the electrons, the electron-atom interaction, and the electron-electron interactions respectively. If we neglect the third term we recover the equation of a Hydrogenic atom which can be solved algebraically.


Our approach is to calculate the the elements of the Hamiltonian matrix in the Hydrogenic basis. We can then explicitly diagonalise the matrix in this basis; if the electron-electron term is small the matrix will be almost-diagonal. I will only cover the case of the bound states; the unbound states do need to be considered at a future point, but at least near the ground state their contribution should be negligible.


The Hydrogenic atom can be simultaneously diagonalised in a number of different basis sets (corresponding to different coordinate systems); we need a basis whose symmetry is preserved by the perturbation. Since the perturbed Hamiltonian is spherically symmetric, we choose the basis H, L, Lz where the bound states are characterised by the quantum numbers n, l, m, s ( $n \geq 1$, $0 \leq l < n$, $|m| \leq l$, $s = \pm \frac{1}{2}$  with corresponding eigenvalues $\frac{Z^2}{2n^2}$, l(l+1), m. The electronic states of an n-electron atom, neglecting electron-electron interactions, are then the antisymmetric tensor products of these states.


We now proceed to calculate the matrix elements of the full Hamiltonian in this basis. The only non-trivial part of the calculation are the terms  $\wedge_{i=1}^{N} \langle n_i, l_i, m_i, s_i | \frac{1}{r_{st}} \wedge_{j=1}^{N} | n_j, l_j, m_j, s_j \rangle$ . The spins and terms i, j not equal to s, t factor through, giving Kronecker deltas. The remaining calculation is  $\langle n_1, l_1, m_1, n_2, l_2, m_2 | \frac{1}{r_{ij}} | n'_1, l'_1, m'_1, n'_2, l'_2, m'_2 \rangle$ . <s>Since the term commutes with  $L_i$  and  $L_j$ , it is also proportional to  $\delta_{l_1}^{l'_1} \delta_{l_2}^{l'_2} \delta_{m_1}^{m'_1} \delta_{m_2}^{m'_2}$.</s> This term commutes with $L_i + L_j$, but not with $L_i$ and $L_j$ separately (intuitively rotating just one of the two electrons will change the distance between them).


Integrate in spherical coordinates over first  $r_1$ , then  $r_2$  setting the z-axis of the second coordinate system along the vector  $r_1$ . Then  $\theta_2$  is the angle between the two electrons at the nucleus, and the integrand is  $\frac{1}{r_{12}} = \frac{1}{\sqrt{r_1^2 + r_2^2 - 2 r_1 r_2 \cos(\theta_2)}}$ , and consequently the first solid angle integral is trivial.


Thus we just need to evaluate  $\int d\Omega Y_{l_2}^{m_2}(\theta, \phi) {Y_{l_2}^{m_2}}^*(\theta, \phi) \int_{0}^{\infty} dr_1 r_1^2 R_{n_1}^{l_1}(r_1) {R_{n'_1}^{l_1}}^*(r_1) \int_{0}^{\infty} dr_2 r_2^2 R_{n_2}^{l_2}(r_2) {R_{n'_2}^{l_2}}^*(r_2) \frac{1}{\sqrt{r_1^2 + r_2^2 - 2 r_1 r_2 \cos(\theta)}}$  (notice that the integral must be invariant under interchange of all 1 labels with 2 labels; in practice we make the choice that makes the integral easiest).


We now separate the  $r_2$  integral into two regions; where it is less than  $r_1$  the  $\frac{1}{r_{12}}$  term can be expanded as  $\sum_{t=0}^{\infty} \frac{1}{r_1} \left( \frac{r_2}{r_1} \right)^t P_t(\cos(\theta))$  where  $P_t$  is a Legendre Polynomial, and in the other region we switch  $r_1$  with  $r_2$ .


The integral then becomes  $\sum_{t=0}^{\infty} \int d\Omega Y^{m_1}_{l_1} (\Omega) Y^{-m_1}_{l_1}(\Omega) \sqrt{\frac{4 \pi}{2t+1}} Y^0_t(\Omega) \int_0^{\infty} dr_1 R_{n_1}^{l_1}(r_1) {R^{l_1}_{n'_1}}^*(r_1) r_1^{2+t} \int_0^\infty dr_2 r_2^{1-t} R_{n_2}^{l_2}(r_2) {R_{n'_2}^{l_2}}^*(r_2)$  plus the integral switching r1 with r2 (after a fiddling change of coordinates).


The angular integral is a combinatorial quantity, which [can be expressed](http://mathworld.wolfram.com/SphericalHarmonic.htm) in terms of the Clebsch-Gordan coefficients. It can be expressed using recurrence relations which can be used to compute this part of the integral. [In fact there is an [explicit ](http://en.wikipedia.org/wiki/Racah_W-coefficient) combinatorial representation, although in practice it would be quicker to compute it using recurrence.]


The inner radial part of the integral can be calculated by expanding the Legendre polynomials as a power series and using the relation  $\int_R^\infty e^{- \alpha r} r^k = \frac{e^{- \alpha R}}{\alpha^{k+1}} \sum_{j=0}^{k} (R \alpha)^j \frac{k!}{j!}$ , and the outer part of the integral can then be calculated using this relation again with R=0. This is simply a combinatorial factor than needs to be determined.


Thus once we have evaluated these combinatorial quantities, and combined them all to get an expression for the matrix elements of the total Hamiltonian H we can truncate it to a finite basis, and then diagonalise it computationally. It is a very interesting question as to how the truncation affects the eigenvalues.