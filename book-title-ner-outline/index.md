---
categories:
- nlp
- ner
- hnbooks
date: '2022-06-20T08:00:00+10:00'
image: /images/hn_books.png
title: 'Side Project Outline: Book Title NER'
---

I'm starting a month long project to extract book titles from Hacker News using Named Entity Recognition.
I've been thinking lately about how Data Science can learn from the practices that have emerged in software development, and wanted to find good books on the subject.
A lot of the ones I'd read, such as Feathers' [Working Effectively with Legacy Code](https://www.goodreads.com/book/show/44919.Working_Effectively_with_Legacy_Code) had come out of Hacker News.
However this is a hard thing to search with using traditional search techniques.

It turns out there's already there's also [MapFilterFold](https://mapfilterfold.com) that searches "Ask HN" threads on Hacker News for book titles.
It sounds like this list is semi-manually curated and results are manually validated.
The site is designed very well; there are categories on the front page, and the detail page has co-recommendations as well as the threads it was recommended in and extracts *specific to that book* (often the thread has lots of other text).
So I can select the "Programming" category, click into the [top result](https://mapfilterfold.com/books/550), "The Pragmatic Programmer", and see it's often recommended with "Clean Code", "Refactoring", "The Effective Engineer", and "Thinking in Systems", and go down to read that people found the "prototype" and "tracer bullet" ideas useful.

There are also sites that extract books using Amazon links; [Hacker News Books](https://hackernewsbooks.com/) and [Books Reddit](https://booksreddit.com/) for HN and Reddit respectively.
This is a high precision strategy, but low recall since most mentions won't have an Amazon link.
For example Martin Fowler's Refactoring gets an entry in Hacker News Books for the [first edition](https://hackernewsbooks.com/book/refactoring-improving-the-design-of-existing-code/8b8771a6459f5e6bc4d89ef236355676) and [second edition](https://hackernewsbooks.com/book/refactoring-improving-the-design-of-existing-code-2nd-edition-addison-wesley-signature-series-fowler/b9c3429e6ed36d327c26140759f0e1ce) with 20 and 5 comments respectively.
Searching for "refactoring fowler" in Hacker News comments currently gives me [260 results](https://hn.algolia.com/?dateRange=all&page=0&prefix=false&query=refactoring%20fowler&sort=byPopularity&type=comment); there's a lot of comments missing.

The real reason I want to do this though is I've been looking for a good Natural Language Processing project and this fits the bill.
The *traditional* way of running this kind of project is to carefully building a model and specifications, writing detailed annotation guidelines, and spending a lot of time manually annotating corpora (or outsourcing to something like Amazon's Mechanical Turk) to train a machine learning model.
This is a lot of upfront planning work to get this pipeline right, and if you discover while training the model that the specification wasn't right you have to throw out a lot of annotations.
Matthew Honnibal, of SpaCy fame, has a [PyData talk](https://www.youtube.com/watch?v=jpWqz85F_4Y) on shortening this cycle and iterating the data and model together.
I'd love to see if I can make this kind of rapid prototyping approach can work.

The first tool in speeding up this process is "weak supervision", popularised by [Snorkel](https://www.snorkel.org/get-started/) in their 2016 [Data Programming paper](https://arxiv.org/abs/1605.07723), of using noisy labelling heuristics to bootstrap a machine learning model.
Amazon links to books is a good high precision heuristic to find a book.
However there are many others like string matching on a big list of book names (a gazetteer), or looking for language patterns like `the book <NNP>` or `<NNP> by <PER>`, or looking for "Ask HN" threads with "Book" in the title.
Then we can aggregate these noisy labels, typically using generative models to estimate the correlation between functions and give a probabilistic label.
These are then used as the input to a machine learning model.
I want to try this with Snorkel and [skweak](https://github.com/NorskRegnesentral/skweak) which generalises to NER using HMM and integrates with SpaCy.

The second tool is reducing the data required using transfer learning.
Until around 5 years ago the standard way of training a NER model was to train a randomly initialised CRF (Conditional Random Field) model, typically with some hand tuned features, at the end of a pipeline consisting of tokenization, prediction Parts of Speech and finding Lemmas using something like [Stanford Core NLP](https://stanfordnlp.github.io/CoreNLP/pipeline.html).
The problem is that you needed a lot of data to train the CRF from scratch and if your text was unlike the "standard texts" (usually news stories) the pipeline may perform badly reducing the quality of the data to the model.
In this case you may need to reannotate parts of speech and tweak the tokenization to improve performance.
An emerging approach is to use pretrained dense vector representations of text, especially using Transformer models, which already represent similar words close together (as opposed to one hot encoding models which needs to learn these relations from the data).
Then a neural network is trained on this representation which, given the pretraining, requires a lot less data to get to a good accuracy.
I'm interested in trying this and want to look into [Stanza](https://stanfordnlp.github.io/stanza/), [flairNLP](https://github.com/flairNLP/flair), [SpaCy](https://spacy.io/universe/project/video-spacys-ner-model) and BERT/RoBERTa models.

The third tool is efficient labelling tools using active learning.
The traditional approach of annotating data is to take each document and label it separately.
However dragging NER spans over text, especially with a lot of categories, is very time consuming for every single example.
Also a lot of these aren't particularly informative; I'm sure I'd end up labelling SICP *many* times when it could be found by a simple model.
A better approach is to correct model predictions which is faster, especially those the model is uncertain about which will give more model improvement.
This approach is called *active learning* and is avaiable in labelling tools such as [Label Studio](https://labelstud.io/) and [Prodigy](https://prodi.gy/).

The other reason this is technically interesting is further techniques could be used to extract more information.
The books and comments could be clustered, or the comments could be summarized, or searched using Question Answering techniques.
These techniques are a lot better performing and more accessible now and I'd love to see how they could be used for effect.

The approach I'm going to take is to quickly deliver an end-to-end baseline and iterate on it.
I could spend a lot of time exploring these different options on a task, but maybe that's not even the right problem.
I want to learn how to use these techniques to create value, and by iteratively building something I can focus on what I think will improve the outcome the most.
This is a learning project, not a business, and the outcome is something I find interesting, but I don't want to spend a whole month trying to optimise an F1-score.

As a starting point I've exported all 2021 Hacker News data from BigQuery [using Kaggle](https://www.kaggle.com/code/edwardjross/hackernews-2021-export/notebook).
I've also found a [large dataset of books on Amazon](https://github.com/uchidalab/book-dataset) and there are many potential datasets for book names such as [open library](https://openlibrary.org/developers/dumps).
The next step will be to do some exploratory data analysis on the data and find the quickest way to get some good book data out.


**Update 2022-09**:
I came across [Hacker News Readings](https://hacker-recommended-books.vercel.app/category/0/all-time/page/0/0) from a [Show HN](https://news.ycombinator.com/item?id=28595967) which apparently does this, but unfortunately there's no code or data available.
It's a good strong baseline though.