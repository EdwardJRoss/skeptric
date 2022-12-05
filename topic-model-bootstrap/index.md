---
categories:
- data
- nlp
date: '2020-08-28T08:00:00+10:00'
image: /images/lda_topic_model.png
title: Topic Modelling to Bootstrap a Classifier
---

Sometimes you want to classify documents, but you don't have an existing classification.
Building a classification that is mutually exclusive and completely exhaustive is actually very hard.
Topic modelling is a great way to quickly get started with a basic classification.

Creating a classification may sound easy until you try to do it.
Think about novels; is a Sherlock Holmes novel a mystery novel or a crime novel (or both)?
Or do we go more granular and call it a detective novel, or even more specifically a [*whodunit*](https://en.wikipedia.org/wiki/Whodunit)?
The answer depends on what you're trying to do; but it's often not obvious from the outset.

Topic modelling is a great way to quickly experiment with different classifications.
In my experience [Gensim's LDA model](https://radimrehurek.com/gensim/auto_examples/tutorials/run_lda.html#sphx-glr-auto-examples-tutorials-run-lda-py) works very well on short and very short texts (think things like survey responses and reviews).
Once you've proved out a basic idea and got it working you can then work towards a more sophisticated classification.

While there may be ways to automatically evaluate the number of topics, it's often easy just to inspect it by hand.
Try a few different numbers of topics and look at the top words and documents in each topic.
If many topics contain a mixture of concepts then try increasing the number of topics.
If many topics contain similar concepts then try decreasing the number of topics.

LDA is a bag of words model, so it's important to normalise words to their root form (e.g. removing inflections).
Sometimes, especially with very short texts, you will get the same theme represented across multiple topics when people use different words to describe the same concept.
This can be handled by building a grouping over the topics, and then summing (or perhaps averaging) the topic scores over the group.

Once you've got to a reasonable output you can label each of the topics (or groups of topics), and you've got an initial classification and classifier.
You can use this to determine whether the model has actual value before you invest in improving the classification, labelling and building a classifier.