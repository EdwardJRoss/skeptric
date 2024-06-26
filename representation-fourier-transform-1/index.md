---
date: 2011-06-29 06:36:29+00:00
image: /images/dft.jpg
title: 'Linear representation of additive groups and the Fourier Transform: Part 1'
---

In this article I will show that the cyclic group of order n, that is the set  $\{0,1,2,\ldots,n-1\}$  under addition modulo n motivates the discrete Fourier transform on a particular finite dimensional complex inner product space, and gives many of its properties. In a subsequent article I will extend this to the general Fourier transform and its relation to the group of integers and real numbers under addition.


<!--more-->


To begin I want to consider linear representations of the cyclic group of order n: that is I want to assign to each element of the group a linear operator on an inner product space in a way consistent with the group structure [or if you prefer, to find a homomorphism from the cyclic group to the group of automorphisms of an inner product space (an orthogonal group)]. There are lots of ways to do this, for lots of different vector spaces – the simplest is to map every group element to the identity (the *trivial (linear) representation*).


It would be nice to have some sort of canonical linear representation. Given a set we can form a vector space by taking all formal linear combinations of its elements (that is we consider the elements of the set to be linearly independent vectors, and the vector space is their span). If a group acts on that set we can extend it to a linear representation of the induced vector space by extending the group linearly; this is called the **permutation representation**.


For example if the set is  $\{a,b,c\}$  the vector space is three dimensional and consists of all elements of the form  $\{x_a a + x_b b + x_c c| x_a,x_b,x_c \in \mathbb{C}\}$ . The group of all permutations on three elements acts on the set, and given such a permutation  $\sigma: \{a,b,c\} \to \{a,b,c\}$  it is represented by the linear mapping  $x_a a + x_b b + x_c c \to x_a \sigma(a) + x_b \sigma(b) + x_c \sigma(c)$ .


Now the group G acts on the set G by left multiplication, and so we can construct a permutation representation. This is called the **regular representation** of G.


What does this look like for a cyclic group of order n? The vector space has a basis of  $\{e_0,e_1,e_2,\ldots,e_{n-1}\}$ , and the group element 1 is represented by the linear transformation S satisfying  $Se_i=e_{i+1}$  (where addition is modulo n). The group element k=1+1+…+1 is represented by  $S^k=SS\cdots S$ .


There is also a natural inner product  $(e_i,e_j) = \delta_{i,j}$  and this is invariant under S (that is S is unitary). As a matrix  $S=\begin{bmatrix}    0 & 0 & 0 & \ldots & 0 & 1\\    1 & 0 & 0 & \ldots & 0 & 0\\    0 & 1 & 0 & \ldots & 0 & 0\\    \vdots &\vdots &\vdots & \ddots &\vdots &\vdots\\    0 & 0 & 0 & \ldots & 1 & 0\end{bmatrix}$ .


Now since S is unitary it is normal and hence by the spectral theorem unitarily diagonalisable. So let’s look for it’s eigenvectors and eigenvalues: since  $S^n = I$  it’s clear its eigenvalues must be nth roots of unity, so denote  $\omega = \exp{2\pi i/n}$  (the choice of sign, and to some extent root, is arbitrary). We can in fact easily see that  $v_k = (e_0 + \omega^{-k} e_1 + \omega^{-2k} e_2 + \ldots \omega^{-(n-1)k} e_{n-1})/\sqrt{n}$  is a normalised eigenvector of S with eigenvalue  $\omega^{k}$  (go on, check it!). Actually the normalised eigenvectors are only determined up to an overall phase, so  $v'_k=e^{i \phi_k} v_k$  would work equally well, but I’ll stick to these phase conventions for convenience.


The diagonalising matrix is then  $F= \frac{1}{\sqrt{n}}\begin{bmatrix}    1 & 1 & 1 & \ldots & 1 \\    1 & \omega & \omega^2 & \ldots & \omega^{n-1} \\    1 & \omega^2 & \omega^4 & \ldots & \omega^{2(n-1)}\\    \vdots &\vdots &\vdots & \ddots &\vdots\\    1 & \omega^{n-1} & \omega^{2(n-1)} & \ldots & \omega^{(n-1)(n-1)}\end{bmatrix}$ .


So  $F^\dagger S F = \mathrm{diag} (1,\omega,\omega^2,\ldots,\omega^{n-1})$ . In fact F diagonalises every group element by multiplication:  $F^{\dagger} S^k F = (F^{\dagger} S F)^k = \mathrm{diag}(1,\omega^k,\omega^{2k},\ldots,\omega^{(n-1)k})$


F is precisely the [discrete Fourier transform](http://en.wikipedia.org/wiki/Discrete_Fourier_transform#Definition) (up to a choice of normalisation): if  $v=\sum_{j=0}^{n-1} v^n e_n$ , then  $F(v) = \frac{1}{\sqrt{n}}\sum_{j,k=0}^{n-1} e^{(-2\pi i/n) j k} v^j e_{k}$ .


Many of the [properties](http://en.wikipedia.org/wiki/Discrete_Fourier_transform#Properties) of the discrete Fourier transform follow immediately; we know it is unitary by the spectral theorem which is precisely the Plancherel theorem. In particular it is invertible, which gives completeness. One half of the shift theorem is also immediate  $FS^k = F S^k F^\dagger F = (F^{\dagger} S^{-k} F) F = \mathrm{diag} (1,\omega^{-k},\omega^{-2k},\ldots,\omega^{-(n-1)k}) F$ . One can see from the explicit form for F that  $F(e_i)=F^{\dagger} (e_{-i})$  and so if we define the operator  $N e_i = e_{-i}$  then  $F^2=F F = F F^{\dagger} N = N$  (though this would be different if we had chosen a different normalisation condition), so applying F to the half of the shift theorem above gives the other half (is there an easier way to see this?).


What about convolutions? Given that each basis vector corresponds to a group element, there is a natural algebraic structure on the vector space, namely  $e_i \otimes e_j = e_{(i+j)}$  (where as usual addition is modulo n). This is precisely a convolution; Excercise: by requiring  $\otimes$  to be distributive and expanding in component prove  $v \otimes w = \sum_{j=0}^{n-1} \sum_{k=0}^{n-1} v^k w^{j-k} e_j$ . What about the convolution theorem? Well we don’t really have an idea of a multiplicative structure (yet) so it doesn’t really make sense.


What is the exact structure on V? There’s an inner product, but there’s also a **relative ordering** of the basis elements; it doesn’t matter where we start numbering the basis elements (except in the definition of convolutions) but S defines an order for them relative to each other. So to say the Fourier transform is defined by a complex inner product space is lying a little, because there is this extra structure. [Also, considering the Fourier transform is only defined up to a phase it could be more natural to think of two vectors being equivalent if they differ only by a phase.] Actually there is a much more natural way to introduce this structure.


There is another way to think of a permutation representation. We form the vector space associated to a set as the vector space of all linear functions from the set to the complex numbers. The basis vector corresponding to the element s is the characteristic function of s,  $\delta_s: S \to \mathbb{C}$  which maps s to 1 and every other element to 0. (Exercise: Show this is equivalent to the description given before, at least if the set is finite). An arbitrary function can be decomposed into the basis of characteristic functions:  $f = \sum_{s \in S} f(s) \delta_s$ . The action of a group element is  $(g \circ f) (s) = \sum_{t \in S} f(t) \delta_{g \circ t} (s) = \sum_{t \in S} f(t) \delta_{t} (g^{-1} \circ s) = f(g^{-1} \circ s)$ .


Now let’s look back at the regular representation of the cyclic group through this lens. We consider functions  $f:\mathbb{Z}/n \to \mathbb{C}$ , with the inner product  $(f,g) = \sum_{m=0}^{n-1} f(m)g(m)$  and we have the shift operator  $S \in \mathrm{Aut}(\mathrm{Map}(\mathbb{Z}/n,\mathbb{C}))$  given by  $(Sf)(m)=f(m-1)$ . The Discrete Fourier Transform is given by  $(Ff)(m) = \frac{1}{\sqrt{n}} \sum_{k=0}^{n-1} f(k) e^{-(2 \pi i/n) m k}$ . The diagonalisation property is that  $F^\dagger S F$  is a multiplicative operator, equivalent to pointwise multiplication by the function  $\hat{S}(m)=e^{2\pi i m/n}$ . (Indeed [Halmos notes](http://www.math.wsu.edu/faculty/watkins/Math502/pdfiles/spectral.pdf) that any normal operator can be unitarily mapped to a multiplicative operator is one way of viewing the spectral theorem).


A convolution is then  $(f \otimes g)(m) = \sum_{k=0}^{n-1} f(k) g(m-k)$ . Now taking the Fourier transform of a convolution of basis elements  $F(\delta_j \otimes \delta_k) = F(\delta_{j+k}) = \sum_{l=1}^{n} \omega^{-(j+k)l}\delta_l$ , and using that the pointwise product  $\delta_l \delta_m = \delta^l_m \delta_l$  (no sum) means we can rewrite it as  $\sum_{l=1}^{n} \sum_{m=1}^{n} \omega^{-jl} \delta_l \omega^{-km} \delta_m$  that is  $F(\delta_j \otimes \delta_k) = F(\delta_j) F(\delta_k)$ . Applying linearity gives one half of the convolution theorem:  $F(f \otimes g) = F(f) F(g)$ . The other half is readily obtained using  $F^2=N$ . Thus the Fourier transform maps the additional ring structure given by pointwise multiplication to the convolution structure given by the regular representation.


So what have we got? We started looking at regular linear representations of the cyclic group, and to change to a basis in which the group operations were diagonal we invented the discrete Fourier transform.


The power in this idea is there are many generalisations. We could have a look at more complicated groups or even more general algebraic structures. The representation theory of cyclic groups is very simple since they are abelian, there’s a lot more involved in trying to diagonalize the representations of non-abelian groups. We could then have other notions of convolutions and Fourier-type transforms. We could also look at mapping to other vector spaces or even to different geometric structures. If instead of constructing vector spaces over the complex numbers we constructed it over finite fields we would get (for the right combination of dimension of the vector space and characteristic of the field) the finite Fourier transform which is important in coding theory. One could also look at what happens to direct sums, tensor products and the like of the regular representations.