---
categories:
- hnbooks
date: '2022-09-16T20:57:56+10:00'
image: /images/hnbook_bad_ner.png
title: How Not to Do Book Named Entity Recognition
---

I'm working on a project to [extract books from Hacker News](/book-title-ner-outline).
I've found some [heuristics to find books](/finding-book-heuristics) and want to train an NER model to recognise the names of books.
However my first attempt didn't work very well and I wanted to do a post-mortem.

I took a weighted sample of data likely to contain book names, based on the analysis of the heuristics.
Using this sample I manually annotated around 500 comments, many of which had book names.
I then trained a SpaCy NER model, randomly splitting training and evaluation, which got 60% precision and 40% recall on the test set.
I then tried using Prodigy's [`ner.teach`](https://prodi.gy/docs/recipes#ner-teach) recipe to do active learning on the full dataset.
However I found quickly there were a lot of false negatives (the precision was much lower than 60%) and most of the examples to annotate had no entities.

The main mistakes I made were:

1. By evaluating on a different distribution I got overly confident results
2. The training distribution didn't have enough examples of comments not containing book mentions
3. SpaCy's NER model has no notion of confidence/probability, so it can't do uncertainty sampling
4. Difficulty of the annotation model

These are all possible to remedy and the rest of this article will go through them one at a time.

# Evaluating on a different distribution gives overly confident results

This is machine learning 101; you want the distribution of your evaluation data to be as close to your training data as possible.
I had already [designed an evaluation](/evaluating-hn-book), but I hadn't carried it out because it was an extra step.
I thought 60% precision was pretty good, especially because the entity boundaries were sometimes fuzzy (see annotation model below).

The main reason I didn't carry out the evaluation is because I didn't know how to do it.
Setting up a script in advance to evaluate the precision on a trained model would have helped here.
I was hoping active learning would help give me an idea of the models performance, but it turns out SpaCy's NER doesn't have a notion of uncertainty making it difficult.


# Training distribution didn't have enough negative samples

Few Hacker News comments are about books, and annotating lots of negative examples is dull and leads to repetition bias (the annotator is more likely to miss real examples).
To ameliorate this I used heuristics to choose posts likely to contain book titles and only annotated them.
However without more negative samples (in particular more diverse negative examples) the model couldn't distinguish book names from other phrases.

We need some way of annotating more negatives examples for the model.
One method would be to only annotate positive predictions of the trained model; this is useful for both for measuring and increasing precision (potentially at the cost of recall).
Another would be to use the heuristics as weak labels and distantly supervise the model; any comments labelled as negative by the heuristics is treated as a true negative.
Taking this a step further we could separately train a classifier to detect items with a title, and automatically label all items below a certain predicted probability as true negatives.


# SpaCy's NER model doesn't expose confidence

SpaCy's NER model doesn't expose a confidence score, it's always 1.
This means we can't do uncertainty sampling which makes it harder to improve the model with active learning (let alone model driven diversity sampling).
The related [StackOverflow thread](https://stackoverflow.com/questions/66490221/spacy-3-confidence-score-on-named-entity-recognition) suggests either find a way to extract it from `beam_ner` (which seems hard to do) or use a [`SpanCategorizer`](https://spacy.io/api/spancategorizer) which is a more general model for categorising spans of text.

I still haven't gotten over the learning curve with SpaCy and prodigy enough to understand the tradeoffs here, or how to wrap my own models conveniently.
The other approach is to train a classifier to predict whether it contains a book title, and use this as a heuristic for active learning; this might help but it's an independent model so it's hard to say, and won't identify cases when there is a book name but it's hard to predict the boundaries.

# Difficulty of the annotation model

Annotation is an active process; it's like a conversation where you really define what you mean by finding good examples.
Active learning helps you get to the interesting examples more quickly, but in general there's a lot of subtlely.

I found many ambiguous cases when I annotated the 500 examples:

* Is a mention of a movie adaptation of "Dune" a book title?
* there's not enough context and I need external knowledge, or context, to tell whether it's a book
* does a series count? does the word "series" belong in the span? (Lord of The Rings, Cradle series, Harry Potter)
* does it count if a book is mentioned obliquely?
* tokenization problems in SpaCy mean a quote or bracket get stuck to the text to annotate; this is especially true when the book title is in a URL (which is tokenized as a single entity)

as an example of the last consider [this comment](https://news.ycombinator.com/item?id=29186253)

> I have read and recommend Andy Tanenbaum's book on operating systems. He's also written a book on networking and one on distributed systems.

This is almost surely talking about [Modern Operating Systems by Andrew Tanenbaum](https://openlibrary.org/works/OL1970688W/Modern_Operating_Systems); but is "operating systems" in this text a book title?

These questions have aspects in what we're trying to achieve, and how we are going to break down the task.
The overall goal is to enable discovering interesting books on a topic and find out about them from comments on Hacker News.
A good intermediate goal is being able to identify what books are being discussed in a comment.
We could do this directly; treat it as a multi-label classification problem, but treating it as a closed problem makes it difficult to add new books, especially books only mentioned once or twice.
Even within a Named Entity Recognition approach we could try to identify *only* book titles, or identify all titles and have a separate step to work out whether it's likely a book.

Examples where I can't answer the question without extra context are going to be *very* difficult for a machine learning model, and suggest breaking it into separate problems.
In this case it may make sense to first classify a `WORK OF ART`, the title of any book or movie, and then separately identify which of these are books (potentially during the linking step).
In this case we could consider a movie adaptation of "Dune" to be a work of art, or the name of a series, or anything that looks like it may be a name.
One tricky issue with Hacker News could be the titles of articles on websites; they are very common and in many senses they're similar to books (they are read, they have an author, they write about things) but they can't be linked to a book.
I'm not sure how I would deal with this.

Tokenization is a separate issue; for cases like brackets and quotes I can just modify the tokenizer.
For names of books in URLs it would be very difficult to tokenize using traditional methods, but potentially possible with learned tokenizers and transformer models.
A pragmatic decision is to ignore them for now, since they're not the most common, and potentially build a specific model to handle them later.

# Summary and Next steps

Based on the learnings I'm going to take the following steps:

* Refine the annotation task to identifying `WORK OF ART`; this has the additional advantage that I can fine tune the existing model on the SpaCy `en_core_web_trf` model (if I start training on GPU)
* Build a Prodigy NER precision recipe, that filters to examples with a span for binary annotation (based on [`ner.teach`](https://github.com/explosion/prodigy-recipes/blob/master/ner/ner_teach.py))
* Train a "Work of Art" classifier based on distant supervision; this can eventually be used to help assess recall, active learning for NER, and provide negative examples.
* Note any tokenization issues I notice, and extend the SpaCy tokenizer as required

Ideally I'd like to directly use active learning on the entity recogniser, but it doesn't seem like SpaCy makes that easy.
In future I'd like to look into either the SpanCategorizer or building on an off-the-shelf token classifier to be able to do both uncertainty and diversity sampling, but it seems like a larger endeavour.
