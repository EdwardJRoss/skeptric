---
categories:
- data
- nlp
date: '2020-07-09T08:00:00+10:00'
image: /images/weak-seq.png
title: Sequential Weak Labelling for NER
---

The traditional way to train an NER model on a new domain is to annotate a whole bunch of data.
Techniques like active learning can speed this up, but especially neural models with random weights require a ton of data.
A more modern approach is to take a large pretrained NER model and fine tune it on your dataset.
This is the approach of [AdaptaBERT](https://github.com/xhan77/AdaptaBERT) ([paper](https://arxiv.org/pdf/1904.02817.pdf)), using BERT.
However this takes a large amount of GPU compute and finicky regularisation techniques to get right.

There is another approach in the middle between: weak labelling.
This is where you take a bunch of sources of labels that are individually not great, such as heuristic rules, or maybe other models and datasets.
Then you aggregate the information together in an *unsupervised* manner to get a new labeller.
This approach lets you apply domain knowledge in a more scalable way than labelling individual examples, and lets you combine multiple sources of information.
It's also effective in practice and part of the open source [Snorkel tool](https://www.snorkel.org).

Snorkel works for classification problems, but in sequential modelling there's a lot of additional useful constraints.
For example you can't follow inside and org to inside a person.
This is the problem solved in [Named Entity Recognition without Labelled Data: A Weak Supervision Approach](https://www.aclweb.org/anthology/2020.acl-main.139.pdf).
They use a Hidden Markov Model with one emission per labelling function, estimating the parameters in an unsupervised manner.

In particular they combined 50 labelling functions including out-of-domain NER models, Gazetteers, heuristic function and document level relations.
The latter includes constraints like "an entity with the same label is likely to have the same type throughout the document", or a long form of an entity (like Kathleen McKeown) is typically followed by a shortened form in the rest of the document (like McKeown).
These were very effective on the news datasets they tested on.

They used a Dirchlet distribution with an informative prior (which is useful on smaller datasets) for the HMM.
They tried simpler distributions and priors and found these to be useful.
This was then used to train a typical neural sequence tagger.

The result was very effective; it works better than AdaptaBERT and is much simpler and quicker to train.

This seems like a really effective technique; using domain knowledge to quickly bootstrap a good NER.
The [source code is available](https://github.com/NorskRegnesentral/weak-supervision-for-NER).