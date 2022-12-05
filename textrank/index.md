---
categories:
- nlp
date: '2022-02-09T08:38:02+11:00'
image: /images/textrank.png
title: TextRank
---

[TextRank](https://web.eecs.umich.edu/~mihalcea/papers/mihalcea.emnlp04.pdf) (Mihalecea and Tarau, 2004) is the idea of using graph ranking algorithms, like [PageRank](http://ilpubs.stanford.edu:8090/422/1/1999-66.pdf), as an unsupurvised way of extracting key units of text from a document.
The interesting part is how they define graphs on the units of text; filtering text units (e.g. words, sentences) to obtain a set of vertices, and using proximity or similarity measures for defining a graph.
They specifically look at the applications of keyword extraction and summarisation.
This is a really useful and versatile idea; and although it requires application specific tuning, is a valuable tool in the unsupervised text toolkit.

The first application is extracting keywords from 500 abstracts of papers from Inspec (scientific and technical papers).
The original dataset was from Hulth's 2003 paper [Improved Automatic Keyword Extraction Given More Linguistic Knowledge](https://aclanthology.org/W03-1028.pdf), and the dataset is on the web (e.g. [here](https://github.com/LIAAD/KeywordExtractor-Datasets#Inspec), [here](https://github.com/boudinfl/ake-datasets/tree/master/datasets/Inspec) and [here](https://github.com/SDuari/Keyword-Extraction-Datasets)).
It consists of small abstracts and human annotated key phrases.
Their TextRank approach is to:

1. filter the tokens of the abstract to just nouns and adjectives as the vertices
2. form a co-occurance graph (unweighted, undirected) based on window of size 2
3. using PageRank to rank the vertices
4. filtering to the top 1/3 of vertices by rank
5. combining any adjacent words in the filtered vertices to get key phrases

This works quite well for this dataset, although there's a lot of choices in here that I suspect are specific to the dataset and task.
The algorithm looks for dense clusters of keywords, and frequent keywords, and this does much better than just frequent keywords alone.
With all this tuning (which unfortunately sounds like it was done on the test set), they get something that works better in terms of precision and F1 than Hulth's supervised approach.

The second application is text summarisation, based on the [Document Understanding Conference 2002 task](https://www-nlpir.nist.gov/projects/duc/guidelines/2002.html) (which you have to [request access for](https://www-nlpir.nist.gov/projects/duc/data.html).
It consists of news articles and manually generated abstracts.
Their approach is to extract *key sentences* by forming a weighted graph of the sentences based on textual similarity between the sentences, rank the sentences using weighted PageRank, and limit the summary by length.
The particular similarity they use is the number of tokens in common, divided by the sum of log lengths of the sentences (perhaps a cousin of Jaccard Distance).

$$ {\rm Similarity}(S_i, S_j) = \frac{\left \vert S_i \cap S_j \right \vert}{\log(\vert S_i \vert) + \log(\vert S_j \vert)} $$

In terms of [ROUGE-1](https://en.wikipedia.org/wiki/ROUGE_(metric)) it ranks about middle of the systems submitted; which is pretty impressive given the simplicity.
In the example it seems to work well because the subject is in the title line and repeated in the most relevant sentences.


These particular TextRank algorithms are available in software; in R there is [textrank](https://cran.r-project.org/web/packages/textrank/vignettes/textrank.html) which covers both tasks, and in Python textacy has [keyword extraction](https://textacy.readthedocs.io/en/0.11.0/api_reference/extract.html#module-textacy.extract.keyterms.textrank) and gensim has textrank [summarisation](https://radimrehurek.com/gensim_3.8.3/summarization/summariser.html).

The real innovation here isn't the particular approaches used, but the idea of using existing graph ranking algorithms for text extraction problems.
There are a large and growing number of ways to find similarity between text units (for example sentence embeddings and contextual embeddings), other behavioural sources of graph information from interactions with documents, and a large number of graph algorithms.
These could be combined in a multitude of ways to solve problems where the information to be extracted is likely to be threaded throughout a document.
Being unsupervised, and able to work on any scale from characters all the way to entire books, makes it a versatile tool.