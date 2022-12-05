---
categories:
- hnbooks
date: '2022-08-20T13:15:00+10:00'
image: /images/displaying-hn-book-comments-html.png
title: Displaying Hacker News Book Comments in HTML
---

I'm currently working on a project [to extract books from Hacker News](/book-title-ner-outline) to help find interesting books.
Having extracted [extracted ASINs from Hacker News posts](/hn-asin) and linked them to [Open Library records](/open-library-isbn-lookup) I have a basic proof of concept.
Now I want to be able to display this information in some webpages to help people find the books.

I ended up building a minimal prototype by manually curating some examples; this let me focus on the design rather than on the technical aspects.
The [landing page](https://htmlpreview.github.io/?https://github.com/EdwardJRoss/bookfinder/blob/1cc3f24a6fa5fde3c78594614dd834c55944ba10/docs/index.html) is a listing of all the books in decreasing number of comments on Hacker News (which here are just wrong estimates); I picked three books I'm familiar with.
This helps find the most popular books; it doesn't solve the problems of finding books on a topic [it's a skateboard prototype](https://blog.crisp.se/2016/01/25/henrikkniberg/making-sense-of-mvp), a step in the right direction.

<figure>
<img src="/images/displaying-hn-book-comments-html-list.png" alt="Listing view of books" width="300px">
<figcaption><a href="https://htmlpreview.github.io/?https://github.com/EdwardJRoss/bookfinder/blob/1cc3f24a6fa5fde3c78594614dd834c55944ba10/docs/index.html">Listing view</a></figcaption>
</figure>

The detail page then contains an extract from Open Library and I extracted the most recent 3 comments using the Hacker News search.
The comment content varies wildly (the Structure and Interpretation of Computer Programs tends to end up in lists of recommendations, The Art of Computer Programming is talked about much more than read, and Pragmatic Programmer is often cited for useful tips), and they're not always useful.
However for these common books there are clearly *lots* of comments, so once we find a way to extract them we can start working out ways to help extract useful information.

The other finding was linking with Open Library was difficult.
All these books have multiple records with different levels of completeness; some works have every edition and author but no covers or description, others have covers and description but are missing an author.
There is going to need to be some level of human curation of the Open Library sources to get good detail pages (picking covers, checking authors, updating descriptions, maybe choosing the relevant editions to link).

Overall this was a useful experience, and I'm glad I did it manually rather than programmatically filling out templates.
The design is ugly and there's only a few examples, but it's enough to give me a feel of what I'm building towards.
The ASIN approach only returned one result per book, so to build a useful example I'm going to need to extract more books.