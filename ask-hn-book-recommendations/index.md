---
categories:
- nlp
- ner
- hnbooks
date: '2022-06-26T14:17:44+10:00'
image: /images/ask_hn_best_books.png
title: Ask HN Book Recommendations
---

I'm working on a project to [extract books from Hacker News](/book-title-ner-outline).
Most HackerNews posts aren't about books, so we need some heuristics to get posts somewhat likely about books.
I've already used [ASINs](/hn-asin) to extract book links to Amazon; another approach is like [MapFilterFold](https://mapfilterfold.com) to use Ask HN threads about books.

[Ask HN](https://news.ycombinator.com/ask) is a kind of post on Hacker News that allows asking questions to the community.
We can't identify them in the dataset but they typically start with "Ask HN" in the title.
Searching for titles matching the regex `^Ask HN.*\b(?:text)?books?\b` gives some of these threads but also threads about discussing books ("How much do you love discussing books with the people who read them?"), writing books ("Is it worth it for a tech startup founder to write a chapter in a book?"), and reading books ("Ask HN: Have you stopped reading books?").
By further refinining to books containing recommendation words `'\b(?:recommend(?:ed)|best|favou?rite|top)\b'` gives almost entirely book recommendation threads (with the ocassional exception such as "Best place to purchase (used) technical books that's not Amazon?").

The top level comments to these threads are almost always book recommendations, and I saved these.

I find it difficult to understand how to make the tradeoff between recall and precision.
These rules could be used as weak labelers, as a seed model for binary classification, or to filter examples to annotate for NER.
For the latter case these high precision methods make sense (to save time skipping texts with nothing to annotate); for other cases maybe lower precision methods make sense.
I need to think more about my strategy before making more of these rules.

If you want to see the detail I've [comitted the exploratory notebook](https://github.com/EdwardJRoss/bookfinder/blob/master/notebooks/0012-ask-hn.ipynb).