---
categories:
- python
date: '2020-08-06T08:00:00+10:00'
image: /images/html2text_doubleemph.png
title: Double emphasis error in html2text
---

I'm trying to find a way of converting HTML to something meaningful for NLP.
The [html2text](https://github.com/Alir3z4/html2text) library converts HTML to markdown, which strips away a lot of the meaningless markup.
I've already resolved an issue with [multiple types of emphasis](/html2text_bi).
However HTML in the wild has all sort of weird edge cases that the library has trouble with.

In this case I found a term that was emphasised twice: `<strong><strong>word</strong></strong>`.
I'm pretty sure for a browser this is just the same as doing it once; `<strong>word</strong>`.
This is likely the result of some strange processing that no one noticed because it makes no visual difference.

Unfortunately html2text doesn't handle this; for each `<strong>` tag it just surrounds the word with two asterisks.
So we get `****word****`, which isn't the correct markdown; it should just be `**word**`.
If I pass this back through a Markdown parser I'll get back something like `<p><strong><em>*word</em></strong>*</p>`.

The annoying thing is I didn't look very hard for this example.
It, and the previous bug, were in the first twenty job ads I found in [Common Crawl](/common-crawl-job-ads).
While I could patch it I feel like these kinds of examples will keep coming up.

I was hoping that html2text could do the heavy lifting of dealing with all the messy HTML, but it seems like I'm going to keep hitting edge cases.
I think the best solution would be to customise the html2text parser to output what I want.
However I'm not entirely sure what I want, and that would be a lot of effort.

I'm going to use it as the basis of a *good enough* solution for now, and wait until there's a practical need to invest more in it.