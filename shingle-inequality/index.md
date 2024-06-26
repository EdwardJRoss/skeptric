---
categories:
- maths
- data
- nlp
date: '2020-04-14T07:36:57+10:00'
image: /images/shingle_inequality.png
title: Jaccard Shingle Inequality
---

Two similar documents are likely to have many similar phrases relative to the number of words in the document.
In particular if you're concerned with plagiarism and copyright, getting the same data through multiple sources, or finding versions of the same document this approach could be useful.
In particular [MinHash](https://en.wikipedia.org/wiki/MinHash) can quickly find pairs of items with a high [Jaccard index](https://en.wikipedia.org/wiki/Jaccard_index), which we can run on sequences of *w* tokens.
A hard question is what's the right number for *w*?
If you use bags (instead of sets) it turns out that increasing the sequence length decreases the Jaccard index (unless it's 1 or 0).

There are lots of measures of similarity between documents such as the [Levenshtein (or Edit) Distance](https://en.wikipedia.org/wiki/Levenshtein_distance).
However few are efficient at finding similar documents at large scale.
The [MinHash](https://en.wikipedia.org/wiki/MinHash) algorithm allow finding sets that have a high Jaccard index (i.e. number of items in common relative to total number of items) and containment (number of items in common relative to items in one set) efficiently.
Treating a document as a set or bag of words may not be ideal; two long documents on a similar topic may have a lot of words in common without having a similar source.
However if the documents have many long sequences of words in common then they probably do have a similar source.
We can form [n-grams](https://en.wikipedia.org/wiki/N-gram) (also called [w-shingles](https://en.wikipedia.org/wiki/W-shingling)); treating the documents as made up of sequences of words of a fixed length, and then calculate the Jaccard index or containment.
How does changing the size of the sequence effect the scores?

Increasing the shingling length will always decrease the Jaccard index and containment *for a bag* (a.k.a multiset), but not for a set.

# Sets and Bags

A *set* is a collection of distinct elements.
A *bag* (or multiset) is a collection of elements that can be the same.
A bag can be represented as a set by adding an index of the time it occurs.

For example [from the MinHash paper](https://www.cs.princeton.edu/courses/archive/spring13/cos598C/broder97resemblance.pdf) the phrase "a rose is a rose is a rose" contains the 4-grams ["a rose is a", "rose is a rose", "is a rose is", "a rose is a", "rose is a rose"].

As a set the 4-grams are the unordered distinct terms ["a rose is a", "rose is a rose", "is a rose is"].
As a bag we can tread it as the set [("a rose is a", 1), ("rose is a rose", 1), ("is a rose is", 1), ("a rose is a", 2), ("rose is a rose", 2)].
Then we can calculate unions and intersections between bags the same way as sets and the Jaccard index $J(A, B) = \frac{\lvert A \cap B \rvert}{\lvert A \cup B \rvert}$ and containment $C(A, B) = \frac{\lvert A \cap B \rvert}{\lvert A \rvert}$ are well defined.

Another useful representation for bags are as mappings from the item to its frequency (which is 0 if the item is not in the bag).
So in this case it would be `{"a rose is a": 2, "rose is a rose": 2, "is a rose is": 1}`.
Then the union of two bags is the *elementwise maximum*, the intersection is the *elementwise minimum* and the cardinality is the sum of the frequencies.
This gives exactly the same behaviour as treating the bag as a set.
A set can also be seen as the special case of a bag where the frequencies are just 0 or 1.

This gives an intuitive proof of the [inclusion-exclusion principle](https://en.wikipedia.org/wiki/Inclusion%E2%80%93exclusion_principle): $\lvert A \rvert + \lvert B \rvert = \lvert A \cup B \rvert + \lvert A \cap B \rvert$.
For each possible element of *A* and *B* exactly one is the minimum frequency, which can be matched to $A \cap B$, and one is the maximum, which can be matched to $A \cup B$.
Summing over the frequencies of all possible elements the equality must also hold, since it holds for any individual element.

# Increasing shingle length decreases Jaccard index for bags

Intuitively it makes sense that two documents are much more likely to have more words in common than sequences of 2-words, and so as the shingle length increases the Jaccard index should decrease.
In fact this is always the case for bags, but not for sets.

Consider the two documents `S = dog cat` and `T = dog cat lemur`.
Then their Jaccard similarity at shingle length 1 is 2/3, and at length 2 is 1/2.

Now consider the documents `S = dog dog cat cat dog` and `T = dog dog cat cat dog lemur`.
As sets they have the same Jaccard similarity at single length 1; 2/3.
The 2-shingles are `S = (dog, dog), (dog, cat), (cat, cat), (cat, dog)` and T additionally contains `(dog, lemur)`.
So the 2-shingle Jaccard similarity is 4/5, which is greater than 2/3.
However as bags the Jaccard similarity at shingle length 1 is 5/6 (since there are 5 words the same), which is less than 2/3.

Denote the bag Jaccard similarity of a sequence at shingle length *k* as $J_k$ (when *k* is larger than both of the documents we take the similarity to be 0).
Then $J_{k+1}(S, T) <= J_{k}(S, T)$ for any two documents *S* and *T* and shingle length *k*.
The rest of this section will prove this proposition.

## Cardinality of Shingles

The *k* shingles are all subsequences of length *k*.
One way to think of this is by going through each index of the sequence, getting a string of length *k* until you get to the last *k*.
In Python code:

```python
def seq_ngrams(xs, k):
    return [xs[i:i+k] for i in range(len(xs)-k+1)]
```

The number of possible shingles is *N + 1 - k* where *N* is the length of the sequence.
So there are *N* possible 1-shingles (tokens), down to 1 possible N shingle (the whole sequence).
In particular $\lvert S_{k} \rvert = \lvert S_{k+1} \rvert + 1$.

## Inequality of Intersection Cardinality

Consider the elements in $\lvert S_{k+1} \cap T_{k+1} \rvert$.
These are sequences of tokens of length *k+1* in both *S* and *T*.
Suppose that *all* of these elements were part of some long substring in both *S* and *T*, e.g. $a_1 a_2 \ldots a_m$ where $m - k = \lvert S_{k+1} \cap T_{k+1} \rvert$.
Then each subsequence of length *k* must also be in $\lvert S_{k} \cap T_{k} \rvert$, and so $\lvert S_{k} \cap T_{k} \rvert \geq \lvert S_{k+1} \cap T_{k+1} \rvert + 1$.

In fact if they are *not* all in the same subsequence then there will be even more length *k* subsequences in common. 
So in **general** it is true (for bags) $\lvert S_{k} \cap T_{k} \rvert \geq \lvert S_{k+1} \cap T_{k+1} \rvert + 1$.


## Calculating the Jaccard Similarity

By the inclusion-exclusion principle we have:

$$J_k = \frac{\lvert S_k \cap T_k \rvert}{\lvert S_k \rvert + \lvert T_k \rvert - \lvert S_k \cap T_k \rvert}$$ 

Plugging in our previous cardinality equality, the intersection inequality, and the inclusion-exclusion principle gives:

$$J_k \geq \frac{\lvert S_{k+1} \cap T_{k+1} \rvert + 1}{\lvert S_{k+1} \cup T_{k+1} \rvert + 1}$$

Finally note that $\frac{N+1}{U+1} \geq \frac{N}{U}$ with equality if and only if *N* equals *U*.
So $J_k \geq J_{k+1}$ and equality can occur only if the intersection is the same as the union; that is $J_{k+1}$ is 1 or 0.

This also applies to containment: 

$$C_{k} = \frac{\lvert S_{k} \cap T_{k} \rvert}{\lvert S_{k} \rvert} \geq \frac{\lvert S_{k+1} \cap T_{k+1} \rvert + 1}{\lvert S_{k+1} \rvert + 1} \geq C_{k+1}$$

By the same inequality as before $C_{k} \geq C_{k+1}$ where equality is only if $A_k$ is empty or contains $B_k$.

# Implications for shingle size

If you're not sure where to start if you look at length-1 shingles (tokens) you're not likely to miss much.
Then you can increase the shingle length to get a lower falspositive rate (at the risk of increasing chance of false positives).