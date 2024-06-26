---
categories:
- nlp
date: '2021-01-21T20:44:56+11:00'
image: /images/bayes_rule.png
title: Language Models as Classifiers
---

Probabalistic language models can be used directly as a classifier.
I'm not sure if this is a good idea; in particular it seems less efficient than building a classifier, but it's an interesting idea.

A language model can give the probability of a given text under the model.
Suppose we have multiple language models each trained on a distinct corpus representing a class (e.g. genre or author, or even sentiment).
Then we can calculate the probability conditional on that model and compare them to calculate the class.

Concretely we have language models $M_1, \ldots, M_k$ each representing a different class, and we want to assign a class to a text T.
Then using Bayes' Rule we have:

$$\mathbb{P}(M_i \vert T) = \frac{\mathbb{P}(T \vert M_i) \mathbb{P}(M_i)}{\sum_{i=1}^{N} \mathbb{P}(T \vert M_i)}$$

So to calculate the class we find the model that maximised the probability of the text under that language model multiplied by the probability of that model being appropriate.
Assuming that the classes in the training corpora are representative of the data at inference time this would just be the fraction of all data in that class.

In many ways for an N-gram language model this seems similar to naive Bayes, but requires more effort to calculate and I don't believe it would give better results.
But it's something worth considering.