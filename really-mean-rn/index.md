---
date: 2011-05-06 03:07:12+00:00
image: /images/vectors.svg
title: Do you really mean ℝⁿ?
---

In mathematics and physics it is common to talk about  $$\mathbb{R}^n$$  when really we mean something else that can be represented by  $$\mathbb{R}^n$$.


Consider mechanics or geometry, these are often represented as theories in  $$\mathbb{R}^n$$ , but really they don’t occur in a vector space at all! Look around you, a three-dimensional description of space probably seems reasonable, but where’s the origin? [Perhaps the centre of your eyes could be an origin, but someone else would disagree with you]. Classical mechanics, special relativity and geometry are much better described as an **affine space** – which is a vector space without an origin.


<!--more-->


Of course to do any calculations we need to choose an origin in the affine space and choose some axes, and this allows us to represent our space as  $$\mathbb{R}^n$$ . But if we calculate the volume of a box (relative to another box) using two different origins we expect to get the same answer. This is precisely what the affine structure says: our results are independent of our choice of origin (that is they are invariant/covariant with respect to translations).


Now suppose you have a genuine $$n$$-dimensional vector space (perhaps you’ve chosen an origin in your geometry), we can now call this  $$\mathbb{R}^n$$  right? Wrong;  $$\mathbb{R}^n$$  has a natural ordered basis  $$e_1=(1,0,\ldots,0), \ldots, e_n=(0,\ldots,0,1)$$ . Again, taking you as the origin in the space around you – what are the three basis vectors? There are lots of choices!


Alright so let’s choose $$n$$ (ordered) linearly independent basis vectors, by identifying the the $$i^\mathrm{th}$$ one with  $$e_i$$  (for the space around us we could choose (magnetic) North, (magnetic) East and (gravitational) up), and this gives us axes. But there’s nothing super-special about these choices – again our box-volume should be independent of our choice of axes. This is encoded in the vector space structure: there’s no canonical choice of basis, and all physical quantities should be properly covariant with an appropriate change of basis.


Fine --- we’ve taken an $$n$$-dimensional vector space and chosen an ordered basis --- surely we can now call it  $$\mathbb{R}^n$$ ? Well it depends on how much extra structure you have on your vector space. The ordered basis for  $$\mathbb{R}^n$$  gives it a canonical orientation. There is a canonical inner product on  $$\mathbb{R}^n$$ , the dot product  $$\langle e_i, e_j \rangle = \delta_{ij}$$ . (It is exceedingly rare to see the distinction made between  $$\mathbb{R}^n$$  as a vector space, an oriented vector space and an inner product space.) So your vector space must at least have an orientation and an inner product (and your choice of ordered basis must be oriented and orthonormal) – but this is a lot of extra structure to throw in.


A lot can be said for the geometry of affine spaces over a vector space (that is without a metric or orientation). There is no concept of absolute volume (this is given by a metric), but there is a relative notion of **oriented volume** for $$n$$-dimensional parallelograms (which is essentially “how many copies of parallelogram 1 fit into parallelogram 2″ with the conditions that volume is translation invariant and scaling a side by some factor scales the volume by the same factor). In fact we can talk about the relative volume for any hyper-polygon (by cutting it into triangular pieces – a parallelogram is just two triangles stuck together) and approximate the relative volume of other objects such as spheres (this is how a lot of Greek geometry was done). There is also the notion of an **affine transformation**, one that preserves the affine structure (essentially a shift followed by a linear transformation), and a **determinant** which is the amount by which an affine transformation scales the volume of $$n$$-dimensional parallelograms.


If we push even further to locally affine transformations the determinant will naturally generalise to the **Jacobian** of the transformation, and one can define the **exterior derivative**. Or thinking of spaces that are locally affine leads to manifolds. But this is a whole essay in itself.


Of course we do often have more structure – Euclidean geometry, classical mechanics, Electrodynamics and special relativity all have metrics – a notion of absolute volume. But it is interesting to separate the metric independent behaviour (for instance in Electrodynamics the Maxwell equations  $$\mathrm{d} \mathbf{F}=0$$ ) from the metric behaviour ( $$\mathrm{d} \star \mathbf{F} = \mathbf{J}$$ ). Certainly it is useful to write equations in a form where they are explicitly independent of the choice of origin and basis (as opposed to writing them in  $$\mathbb{R}^n$$ ) – this is the advantage of Lagrange’s method of mechanics over Newton’s.


We should explicitly distinguish these structures from $$\mathbb{R}^n$$. A vector space should just be denoted by a letter like V, with possibly a superscript to denote its dimension like  $$V^{(n)}$$ . An affine space could be denoted  $$A^{(n)}$$ , an affine space with a positive definite inner product is called Euclidean space  $$E^{(n)}$$  and an affine space with an inner product signature $$(n-1,1)$$ is called Minkowski $$n$$-space  $$M^{(n)}$$ . I would advocate using  $$E^{(m,n)}$$  for an affine space with an inner product signature $$(m,n)$$.


As I’ve said at the end of the day to do calculations we will go into a specific representation which may look like  $$\mathbb{R}^n$$  and this is fine – but it is important to state and develop the theory in ways that are independent of these choices.


End Note: Given a basis for a vector space V  $$\{e_1,\ldots,e_n\}$$  we can define a **dual basis** for $$V^*$$,  $$\{e^1,\ldots,e^n\}$$  by  $$e^i(e_j) = \delta^i_j$$ . The mapping of a basis onto its dual gives an isomorphism from $$V$$ to $$V^*$$, but this isomorphism isn’t *natural*. This has a specific categorical meaning but roughly its because if we have a linear transformation $$L$$ on $$V$$, $$L$$ has a natural action on $$V^*$$. In terms of the matrix of the transformation the action on $$V^*$$ is the adjoint matrix (that is the transpose or Hermitian conjugate depending if the matrix is real or complex). But the dual basis transforms by the *inverse* to maintain  $$e^i(e_j) = \delta^i_j$$ . So the adjoint action preserves the identification of bases (and so the isomorphism is independent of our choice of basis) if and only if  $$T^* = T^{-1}$$ , which means the matrix is orthogonal or unitary. However apriori there is no reason to only use this type of linear transformation so the identification is not canonical. If, however, there is a metric on our vector space we should only use linear transformations that preserve this metric, and in terms of an orthonormal basis, this means we only use orthogonal/unitary transformations. So a metric induces a canonical isomorphism between a vector space and its dual! It is not too hard to see conversely that an isomorphism between a vector space and its dual gives a metric.


This is another reason (related to the existence of a natural inner product) a vector space should not be represented by  $$\mathbb{R}^n$$  – it is canonically identified with its dual.