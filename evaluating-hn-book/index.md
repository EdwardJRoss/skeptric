---
categories:
- nlp
- ner
- hnbooks
date: '2022-06-29T04:53:58+10:00'
image: /images/train_test_split.png
title: Evaluating Book Retrieval from Hacker News
---

I'm working on a project to [extract books from Hacker News](/book-title-ner-outline).
I've been thinking about ways to bootstrap this process such as transfer learning, weak labelling, and active learning.
I was reading Robert Monarch's excellent book [Human-in-the-Loop Machine Learning](https://www.manning.com/books/human-in-the-loop-machine-learning) where he gives good reasons as to why you should start with a separate random dataset for evaluation; you want something that is representative of the real distribution, and any use of a model inherits the biases of that model.

However annotating a random dataset for evaluating finding book titles in Hacker News posts would be a terrible experience.
I would be surprised if one in one thousand Hacker News posts contained a reference to a book.
So we'd need to annotate *heaps* of examples, almost all of them negative, to get a few examples for evaluation.
We need quite a few positive examples to get reliable estimates of accuracy, since we need to find the entity, extract it and link it to a record which could be much easier for some kinds of mentions than others.

We actually don't want to measure this kind of accuracy for this use case.
To be useful our system has to have high precision; if we say a book is mentioned in a comment we really want that book to have been mentioned.
This enables us to say these books are really what are being talked about, allows meaningful link backs to comments, and gives trust in the system.
Recall is somewhat important; we don't want to be too biased towards certain kinds of books, and we want to get as many books as we can to give better insight.
But at the end of the day if we miss some examples it's fine.

In this case I'm thinking of implementing a two-pronged strategy for evaluation; one for precision and another for recall.
We start with a separate unlabelled test split on the root post of the thread; for example randomly splitting these by id.
For precision whenever we evaluate a model we pick completely at random subset of positive predictions and label the unlabelled ones; this gives a reasonable estimate of precision.
Over time this set also gives a rough estimate of recall as we get more labelled likely positive predictions.
We can supplement this with a random sample of likely positives using heuristics like [links to Amazon](/hn-asin) or top level comments in [Hacker News Book Recommendation threads](/ask-hn-book-recommendations).
These will give a better measure of recall, especially if we don't make this information available to the model (e.g. removing Amazon links, and not giving the parent information).
If we want to label entities this can be sped up by correcting a pretrained model (like an [Work of Art NER model](/book-ner-work-of-art) or a [Question Answering Model](/qa-zero-shot-book-ner)), rather than annotating from scratch.

This may give a completely fair comparison between models, since I'm building up the evaluation set over time and using different evaluations.
But I'm not trying to micro-optimise a model; I'm trying get to a useful result with a minimum of human time.
In this case getting fast feedback on what makes precision increase and roughly how recall is increasing is more valuable than annotating large quantities of data to get more precise estimates.
There's a time cost to evaluating more examples to increase the precision of evaluation, an information gain.
It's only worth investing in that when it's going to [change a decision](/analysis-decision) on what approach to use.
At the start of a project when there's a lot of low-hanging fruit a rough evaluation is often good enough.