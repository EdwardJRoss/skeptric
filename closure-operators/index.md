---
categories:
- maths
date: 2011-03-28 07:37:27+00:00
image: /images/hypercubeorder_binary.svg
title: Closure Operators
---

Often in mathematics there is the idea of taking the **closure** of some elements under a particular operation. For instance if you have several vectors you may take the span of them, if you have a field and want to add the zeros of a particular polynomial you want to consider the corresponding extension field, given a set you often want to construct “the finest topology such that…”.


<!--more-->


These examples are all generated in a similar way; the objects are all sets with additional structure that is preserved by intersections, but not unions. For instance the intersection of subspaces of a vector space is a vector space and the intersection of subfields of a field is a field. Thinking of a topology as a collection of open sets or closed sets, a topology is a subset of the power set that obeys certain properties. Every topology on a set is a subtopology of the discrete topology, and the intersection of subtopologies is a topology (but the union isn’t!).


So in each case we have a universal or greatest set (*my terminology*), a collection of subsets that represent all subobjects of the universal set (vector subspaces, subfields, subtopologies) that is closed under intersection and includes the universal set. The closure is then a mapping from all subsets of the universal set to the subobjects; it maps a set to the intersection all closed subobjects containing that set. In a sense it takes a set to the “smallest” subobject.


For example if our universal set is  $R^{n}$  (for some fixed  $n=1,2,\ldots,\aleph_0$ ) then one subset is the point  $\{x\}$  for  $x\neq 0$ . The closure of  $\{x\}$  is the intersection of all vector spaces containing  $\{x\}$ , which is just  $\{\alpha x |\; x \in X\}$ . The closure of the empty set is  $\{0\}$ . The closure of any vector space is the vector space itself. The closure of a collection of vectors is precisely their span.


As a topological example the product topology is the finest topology such that the projection operators are continuous. For each projection we take the inverse image of the topology of the target space, and then take the union over all the projections to get a subset (of the power set of the cartesian product). Then the product topology is precisely the closure of this subset. [N.B. Don’t confuse the closure operator **on** topologies with the topological closure; but they do have a common root since the intersection of arbitrarily many closed sets is closed].


So the closure operator is a useful concept that appears often in mathematics. The rigorous definition is due to Tarski. Given a set a **closure operator** is a map  $\bar{}$  from the power set to itself such that:




1.   $X \subseteq \bar{X}$ .
1.   $X \subseteq Y$  implies  $\bar{X} \subseteq \bar{Y}$ .
1.   $\bar{\bar{X}} = \bar{X}$ .



The closed sets are the the ones such that  $\bar{X}=X$ . This is in fact equivalent to my examples, the closure operator I constructed satisfies these properties. To see the opposite equivalence note




*  The universal set is the power set, which is closed by 1.
*  By 2 and 3 the closure of a set is the same as the intersection of all closed sets containing it.



To find out more about this stuff check out the first chapter of Burris and Sankappanavar’s [“Universal Algebra”](http://www.math.uwaterloo.ca/~snburris/htdocs/ualg.html). It’s free!


It’s also interesting to note the close relation to topology; these form part of Kuratowski’s closure axioms for a topology, but preservation of binary unions and the empty set don’t hold in the examples I gave above.


I’ll leave you with a (loose) connection to logic. A boolean algebra corresponds in a natural way to boolean logic, and a Heyting algebra (or Brouwerian lattice) corresponds to intuitionistic logic. The “dual”, in some sense, of a closure operator is an interior operator – if we can take arbitrary unions we define the interior of a set as the union of all *subobjects* contained within the set. A set with an interior operator is a complete Heyting algebra (and this close relation is used to define *pointless topology* – that is topology just in terms of sets and subsets, no mention of points – in terms of the open sets).


Thus a closure operator corresponds to the “dual” of a Heyting algebra. In a Heyting algebra we have the implication  $A \rightarrow B$ , familiar from logic. The dual operation in a dual Heyting Algebra is more like set subtraction  $A - B$ . I’m not sure what the logical structure (if any) corresponding to a dual Heyting algebra is.