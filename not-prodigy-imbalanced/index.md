---
categories:
- hnbooks
- annotation
date: '2022-10-11T18:40:33+11:00'
draft: true
image: /images/
title: How not to use Prodigy's active learning for imbalanced classification
---

I'm working on a project to [extract books from Hacker News](/book-title-ner-outline).
I've already tried [training an NER model](/how-not-book-ner) based on [heuristics to find books](/finding-book-heuristics), but it didn't do well with out-of-domain data creating too many false positives to be useful.
This time I tried using [Prodigy](https://prodi.gy/)'s active learning to pick useful examples.

Unfortunately I found that Prodigy's model of active learning wasn't great for this kind of problem.
In particular due to the extreme class imbalance (maybe 1 in 5000 posts contains the title of a book) a lot of examples need to be rejected to find data worth annotating.
The built-in active learning strategies in Prodigy involve taking the most uncertain predictions in a batch, which most of the time won't get interesting examples to annotate (since most are obviously False).
An alternative strategy is to filter to the most interesting examples on the fly, but this requires a lot of computation to find interesting examples which can slow the annotation process.

For such an imbalanced dataset rather than trying to use active learning on the stream, I would run a batch process to find examples to annotate before each round of annotation.
The rest of this article dives deep into what I tried and the issues I ran into.

# Filtering Named Entity Recognition

Book titles are already in Ontonotes under the broader category of *Work of Art*, which can be domain adapted to this dataset.
To increase the precision I could focus on correcting erroneously predicted Work of Art entities.
To do this I adapted the Prodigy's [ner.teach recipe](https://github.com/explosion/prodigy-recipes/blob/06e5e82f9e649dae5207e6ff0e847134b3b2a836/ner/ner_teach.py) to filter to only examples where there is an entity, which I called [`ner.precise`](https://github.com/EdwardJRoss/bookfinder/blob/38a6d71a299f391178817d86bcbf182ef34359c2/bookfinder/recipe.py).

Unfortunately I found it tricky to get a good annotation flow.
From my previous analysis of [Work of Art to find books](/book-ner-work-of-art) only the `en_core_web_trf` transformer model was good enough to use.
This model is slow to run on a CPU, especially if we need to run on thousands of examples for a single annotation batch.
I spent a long time to get it running on a cloud GPU provider, and it runs reasonably fast on an RTX-5000 (once I injected [`spacy.require_gpu`](https://spacy.io/api/top-level#spacy.require_gpu) into the recipe).
Between each batch there is a pause of around 5-10 seconds, which is just tolerable for annotation (and for some reason Prodigy sometimes repeated the same example in different batches).

At this point I started to get some annotated data, but wasn't familiar enough with SpaCy to understand how to fine tune it.
So I started on a simpler problem; training a classifier to detect whether a post contains a mention of a book.

# Imbalanced Classifier

SpaCy already has a [textcat.teach](https://prodi.gy/docs/recipes#textcat-teach) recipe for updating a trained text classifier.
I thought an interesting approach would be to train an initial classifier from an existing heuristic, and use this as a seed model for active learning.

I found SpaCy's text classification to be clunky to experiment with and really broke my flow, and I spent a long time learning how to work with it and debugging it.

It took me some time to work through all the issues to train a SpaCy classifier.
It expects data in a `.spacy` file from a [`DocBin`](https://spacy.io/api/docbin/) object.
This requires running all the text through a SpaCy (blank) `Language` and annotating each Doc with the class (I found [`nlp.pipe`](https://spacy.io/api/language#pipe) with `as_tuples` to be useful here), then passing them into a `DocBin` to serialise to disk.
`DocBin` requires to load all the data into memory rather than streaming lazily and I ran into some memory issues saving a large sample, and had to reduce my sample size.
I then generated a config with `spacy init` and trained the model.

I really disliked this workflow because doing something like Prodigy's [train-curve](https://prodi.gy/docs/recipes#train-curve) to see how the accuracy increases with data seems to require running this whole process multiple times for each dataset and putting all these separate files on disk, then looping (in the shell?) the `spacy train` commands.

The first time I tried to train an "accurate" model on CPU (which uses word vectors from `en_core_web_lg`) the process would always be killed with Out of Memory.
It took me a while to work out that I needed to reduce the `batch_size` in the config file (the default of 1000 took well over 10GB of memory).
Eventually I managed to train a model that was making some reasonable predictions (which took a while).

The model was fast enough to update in the loop on CPU, however most of the examples weren't interesting.
The active learning strategy uses `prefer_uncertain` which [picks questions that are more than one standard deviation above the average uncertainty](https://support.prodi.gy/t/prefer-uncertain-how-does-it-use-the-stream-to-pick-examples-to-score/131/2) using an exponential moving average.
Unfortunately because the distribution of uncertainty is long tailed in this imbalanced classification task the estimated moving average uncertainty is much too low.
An alternative is to use the `probability` algorithm which randomly drops examples with probability proportional to their distance to 0.5.
This could work if we recalibrated the classifier probabilities appropriately, but the models are so opaque I wouldn't know where to start with it (and even then it may take a while because it will need to filter many examples).

# Next Steps

Getting an active learning strategy right is hard, and I hoped too much the tools could do it for me.
Prodigy's on-demand data streaming approach doesn't work well for this very imbalanced data scenario, and the SpaCy models are quite opaque and require a break in my experimentation flow to train making iteration cycles long.

My approach will now be to update the model and sample data in batches rather than in the loop.
This lets me make predictions all the unlabelled data in one big slow batch process, and then decide how to select appropriate items for annotation.
I'll also switch away from SpaCy models for more generic models (likely transformers) that are more transparent and easier to experiment with.

I'm still not completely clear on how I'll sample the data, but I'll follow some of the advice from the [Human-in-the-Loop Machine Learning Book](book-review-human-loop-machine-learning/) and use a mixture of:

* The most uncertain predictions
* Data very different to what has been seen
* Completely at random

One option would be to adapt the [code](https://github.com/rmunro/pytorch_active_learning) from the [disaster annotation](/disaster-annotation) example from that book.
