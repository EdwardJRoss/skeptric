---
categories:
- data
- nlp
date: '2020-07-06T08:00:00+10:00'
image: /images/job_html.png
title: Using HTML in NLP
---

Many documents available on the web have *meaningful* markup.
Headers, paragraph breaks, links, emphasis and lists all change the meaning of the text.
A common way to deal with HTML documents in NLP is to strip away all the markup, e.g. with [Beautiful Soup's `.get_text`](https://www.crummy.com/software/BeautifulSoup/).
This is fine for a bag of words approach, but for more structured text extraction or language model this seems like throwing away a lot of information.
Is there a better way to process the text while retaining the meaningful information?

A lot of things that you might find in HTML pages aren't useful for NLP; like Javascript scripts, HTML tag attributes specific to the markup, and CSS.
These you almost certainly do want to strip out.
We need to find the right balance between the two of them.

What is meaningful is something close to [Markdown](https://daringfireball.net/projects/markdown/).
At a text span level it supports links, two kinds of emphasis (`em` and `strong`), and inline code (which is less relevant in general text).
At a block level it supports paragraphs, 6 levels of headers, lists (ordered and unordered), quote and code blocks and horizontal rules.
This is likely much more nuanced than is typically used but it's a reasonable compromise for generic internet content.

It's possible to extract Markdown from HTML with [html2text](https://github.com/Alir3z4/html2text) or the excellent [Pandoc](https://pandoc.org/).
Depending on the downstream task you will probably want to use the markup in different ways.
You could customise the approach from html2text, transform with custom Pandoc filters or write a bespoke parser with [html5lib](https://html5lib.readthedocs.io/en/latest/index.html).

At minimum you could try to capture some of the meaning of the markup in plain text; similar to Pandoc's `plain` output.
For example a list could be turned into a semicolon separated run-on sentence, maybe emphasised text should be in caps, certainly paragraphs and line breaks should be separated by newlines.
The target may depend on what you're trying to do, but should produce a better result than just stripping away tags with Beautiful Soup `.get_text`.

For language modelling you could encode the markup as special tokens.
For a rules based tokenizer there are a number of straightforward ways to do this.
An example for emphasis we could use HTML-esque `<em>` and `</em>` tokens.
Or we could precede every token in the span with `<em>`.
We could even do something strange like precede the whole section by `<em> 5` where 5 is the number of tokens in the span.
Typically it's only whole words that are emphasised; if only part of a token is you would have to decide whether to use that information or discard it (depending on how often it occurs and how meaningful it is).
It would be worth experimenting to see what works best for a language model.

For an unsupervised tokenizer like [sentencepiece](https://github.com/google/sentencepiece) you could try to use Markdown or HTML tags directly.
Just try to make it as easy as the tokenizer as possible to capture the information.

Another way to capture the information would be to annotate it on tokenized text.
For example you could mark it up like [Spacy spans](https://spacy.io/api/span).
But you'd need some customised algorithms that could make use of the information.

I still need to experiment more on how to get the most out of text encoded in HTML, but I definitely think there are useful ways to use the information depending on the application.