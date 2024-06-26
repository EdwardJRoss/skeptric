---
categories:
- nlp
- ner
- hnbooks
date: '2022-07-01T08:00:00+10:00'
image: /images/open_library_the_outsider.png
title: 'Open Library: A Book Knowledge Base'
---

I'm working on a project to [extract books from Hacker News](/book-title-ner-outline).
Once I have extracted book titles (e.g. with [NER](/book-ner-work-of-art) or [Question Answering](/qa-zero-shot-book-ner)) I need a way to disambiguate them to an entity, and potentially link them to other information.
The Internet Archive's [Open Library](https://openlibrary.org/) looks like a very good way to do that.

Book titles can be ambiguous and so we need some way to link it to a unique entity.
For example "The Stranger" could refer to one of many books; such as that by Richard Wright, Stephen King, or Colin Wilson.
Books can also have multiple names; "L'Étranger" by Albert Camus may also be known as "The Stranger" or "The Outsider" from different translations.
While we're at it are different translations the same "book"?

Open Library describes itself as "a universal catalog for book metadata".
It's open for the public to edit, they make the data public, and it has over 25 million *works* (books), including most I can think of.
A *work* which can have multiple *edition*s; from their [editing FAQ](https://openlibrary.org/help/faq/editing#work-edition):

> Work and edition are both bibliographic terms for referring to a book. 
> A "work" is the top-level of a book's Open Library record. 
> An "edition" is each different version of that book over time. 
> So, Huckleberry Finn is a work, but a Spanish translation of it published in 1934 is one edition of that work. 
> We aspire to have one work record for every book and then, listed under it, many edition records of that work.

This makes a good fit for our purpose of identifying books.
They also release [bulk data dumps](https://openlibrary.org/developers/dumps) which allow batch processing of our extracts, and include useful metadata such as the ISBN of editions.

For usage in Hacker News we may need to make our own extensions to this.
There are specific references like [SICP](https://hn.algolia.com/?dateRange=all&page=0&prefix=false&query=SICP&sort=byPopularity&type=all) for the *Structure and Interpretation of Computer Programming*, or [The Dragon Book](https://hn.algolia.com/?dateRange=all&page=0&prefix=true&query=Dragon%20Book&sort=byPopularity&type=all) for *Compilers: Principles, Techniques, and Tools* which may not fit into Open Library, but are important for this usecase.
But it's a very good starting point.

I'll need to do some more work to process the data dumps to allow efficient linking.
They're about 10GB of gzipped TSV, including a JSON column.
One approach is the [Libraries Hacked repository](https://github.com/LibrariesHacked/openlibrary-search) which imports it into a PostgreSQL database.