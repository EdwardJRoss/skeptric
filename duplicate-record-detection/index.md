---
categories:
- data
date: '2022-09-14T18:49:59+10:00'
image: /images/duplicate_open_library.png
title: Duplicate Record Detection in Tabular Data
---

How do you deal with near duplicate data, or join two datasets with some errors?
For example when [a book is added to Open Library](/adding-open-library) it's easy for accidental duplicates to occur, and there are many in practice.
There are often small differences between duplicates, such as abbreviating author's names, subtitles, omitting the words like "the" and "a".
Another related problem is joining to another database; for example book and author names extracted from text.
How can we do this?

There's a whole body of research dedicated to this problem, variously known as *record linkage*, *field matching*, *data integration*, *duplicate detection*, *entity resolution*, and about a dozen other things.
A great introduction is [Duplicate Record Detection: A Survey](https://www.cs.purdue.edu/homes/ake/pub/TKDE-0240-0605-1.pdf) by Emlagarmid, Ipeirotis, and Verykios.
There are effectively 4 steps:

1. Data Preparation: Parsing fields and converting them into a standard format to make them as similar as possible
2. Indexing: Efficiently finding pairs of records that are likely to contain matches
3. Similarity Metrics: Measures for how similar two fields are; typical data input errors should give similar scores
4. Classification: Classifying records as duplicate based on the similarity metrics

A good tool for this is the [Python Record Linkage Toolkit](https://recordlinkage.readthedocs.io/en/latest/), providing that you can fit all your data into Pandas data frames.
It has inbuilt methods for each of the steps and is able to run it all together, along with evaluation on standard datasets.

The data preparation steps depend a lot on the data and so they often get the least attention in the literature.
However if you know certain kinds of errors are common, like names being abbreviated or differences in casing or switching names, creating the appropriate columns makes the following steps much easier.
One technique is phoenetic encoding of string columns such as Double Metaphone and Soundex to make phoenetic variations more similar; for example the double metaphone enocding of both "Chebyshev" and "Tchebysheff" is `XPXF`.
Less data preparation requires better choices in the following steps.

Indexing is important because comparing all pairs for duplication has quadratic run-time complexity.
The most common technique is called *blocking*; separate the records into disjoint buckets (say by the first 3 letters of the surname) and only compare the records in the same bucket.
The run-time complexity is then the square of the size of the largest buckets; picking an evenly-distributed buckets produces the best results.
However duplicate records may occur in difference blocks, a common technique is to compare all records across a few different blocking strategies.
There are more indexing methods, as covered in [A Survey of Indexing Techniques for Scalable Record Linkage and Deduplication](http://users.cecs.anu.edu.au/~Peter.Christen/publications/christen2011indexing.pdf) by Peter Christen, but none are clearly better than simple blocking.
Surprisingly I haven't seen Locality Sensitive Hashing mentioned as a blocking strategy (I found it useful for finding [near duplicate job ads](/near-duplicate-review)); it works well with q-grams of characters with Jaccard similarity or vector embeddings using cosine similarity.

The similarity metrics give scores of how similar two records are.
There are many specific algorithms for finding the common types of errors in text.
Character based similarity metrics like [Jaro-Winkler Distance](https://en.wikipedia.org/wiki/Jaro%E2%80%93Winkler_distance) and [Smith-Waterman Distance](https://cs.stanford.edu/people/eroberts/courses/soco/projects/computers-and-the-hgp/smith_waterman.html) are often used.
Knowing the types of errors in the data the metrics can be tuned appropriately, for example by putting lower weights on an edit distance (casing would be a good example).
An alternative is bag of token approaches like q-grams, potentially weighted by TF-IDF, and as mentioned above this can be used for blocking too using Locality Sensitive Hashing.
At the end of this we have some vector of differences accross the field that we need to classify.

The classification step takes the vector of differences and returns whether they are duplicates (or how likely they are to be duplicate).
Here we can use the usual toolbox of machine learning (depending on how much labelled data we have or can acquire), including hard coded rules.
The transitivity property means that if A is a duplicate of B and B is a duplicate of C then A is a duplicate of C; this may require some post-processing to decide which of these clusters to use (for joining across de-duplicated datasets often only the best match should be used).
There are some advanced methods that take transitivity into account, such as [A Bayesian Approach to Graphical Record Linkage and De-duplication](https://arxiv.org/abs/1312.4645).

If you want to know more about this the Python dedupe library has [good documentation of its process](https://docs.dedupe.io/en/latest/how-it-works/How-it-works.html), the US Cenus Beaureau has an [Overview of record linkage and current research directions](https://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.79.1519), and Peter Christen has a whole book on [Data Matching](https://link.springer.com/book/10.1007/978-3-642-31164-2).
Although there's a lot of decisions, it seems like in many cases understanding your data and choosing the right transformations and heuristics will get you much further than advanced techniques; you could build a reasonable solution entirely in SQL (though you probably shouldn't).