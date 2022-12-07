---
categories:
- nlp
- ner
- hnbooks
date: '2022-06-27T21:43:50+10:00'
image: /images/work_of_art_ner.png
title: Book NER as a Work of Art
---

I'm working on a project to [extract books from Hacker News](/book-title-ner-outline).
I've [previously found](/ask-hn-book-recommendations) book recommendations for Ask HN Books.
Now I want a way to extract the book titles and authors.
The Ontonotes corpus contains an NER category called Work of Art (for titles of books, songs, etc.) (see the [PDF release notes for details](https://catalog.ldc.upenn.edu/docs/LDC2013T19/OntoNotes-Release-5.0.pdf)).
I wanted to see how well this worked.

I quickly tried 3 well known systems all trained on this corpus; SpaCy, Stanza, and Flair NLP.
SpaCy (`en_core_web_trf`; the smaller models don't work) and Flair NLP both performed quite well.
They both got a lot of the titles and persons, although struggled on punctuation.
The lack of periods seemed to hurt both the models in hyphen lists, and things like quotations seemed to leak into results.
Stanza performed much worse than the other two models and isn't worth considering.

These aren't good enough to solve the problem well by themselves, but are a good starting point for training a better NER model for getting book titles.

If you want the quick qualitative analyses see the notebooks for [SpaCy](https://github.com/EdwardJRoss/bookfinder/blob/master/notebooks/0022-spacy-ner.ipynb), [Flair](https://github.com/EdwardJRoss/bookfinder/blob/master/notebooks/0023-flair-ner.ipynb), and [Stanza](https://github.com/EdwardJRoss/bookfinder/blob/master/notebooks/0024-stanza-ner.ipynb).