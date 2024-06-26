---
categories:
- hnbooks
- nlp
- ner
- python
date: '2022-07-23T08:00:00+10:00'
image: /images/open_library_edition_metadata.png
title: What's in Open Library Data
---

I'm working on a project to [extract books from Hacker News](/book-title-ner-outline), and want to link the books to records from [Open Library](/open-library).
I've already looked at the process of [adding a book to Open Library](/adding-open-library) and [loading a data export into sqlite](/openlibrary-sqlite).
Now I really want to look through the data and see what's inside.

I do this through two different perspectives; summarising the metadata and looking at some specific records.

# Summarising the Metadata

Looking through a random 1% sample of the works, and the related authors and editions, I analysed the fields that occur more than 1% of the time in the metadata.
I used a sample so the whole thing would fit into memory and make analysis faster.
You can see the details in the [Jupyter Notebook](https://github.com/EdwardJRoss/bookfinder/blob/master/notebooks/0032-openlibrary-eda.ipynb).

It seems like there are a few different processes for adding entities to Open Library.
As well as the public adding and editing fields there seem to be some bulk imports from other book databases, and potentially some other programmatic edits.
In fact around 70% of the editions were imported from an external source, leaving 30% that may have been manually uploaded.
This just reports high level statistics, but it would be interesting to understand the field usage by source.

## Editions

* 70% of the editions have been automatically imported from one of MARC, Better World Books, Internet Archive, or Amazon (listed in `source_records`).
* 64% of editions have at least one ISBN 10 or ISBN 13 (this is asked for in manual uploads, or an LCCN)
* Almost always have a `title`, and sometimes a `subtitle` (41%) a `full_title` (19%; often the concatenation of the `title` and `subtitle`), an `edition_name` (14%), and occasionally `other_titles` (8%).
* Generally have `authors` (89%), and sometimes a `by_statement` (43%) which is how the authors are listed as text in the book
* Editions often contain a `publisher` (96%), `publish_date` (98%) and 58% of the time locations in `publish_places` and `publish_country` (the latter of which is often a US state and not in the Open Library user interface)
* 60% of the time `subjects` are available; other details about what the book is about are available less than 10% of the time, such as `subject_places`, `subject_people`, `subject_time`,  and `genres`
* 48% have `lc_classifications` and 18% a `dewey_decimal_class` which help identify the topic
* It can be connected to other databases using things like `lccn`, `oclc_numbers`, `identifiers`, `ocaid`.
* Sometimes more information is in `notes`, a `table_of_contents`, `description`, or a `first_sentence`
* There are sometimes a `cover` (33%) image hosted on Open Library

## Works

A work can contain multiple editions.
In the Open Libary user interface it's not very clear how you edit a work, but some changes on editions automatically changed the work (such as adding a cover0.

* Works largely have a subset of the fields of editions, not always consistent with the editions
* The authors are normally a superset of the authors of the editions, typically there's only one author (89% of the time), and 5% of the time no author. 
* On average there's 1.3 editions per work

## Authors

Authors are a bit

* On average there's 1.1 works per author
* Most authors have a `name`, 62% a `personal_name`, and 4% `alternate_names`. They're often inconsistent with format (e.g. `Surname, Firstname` or `Firstname Surname`)
* 22% have a `birth_date` and 4% `death_date` (free text) which could be useful for disambiguation
* 7% have `remote_ids` linking to wikidata, VIAF and ISNI where additional information can be obtained
* Less than 2% have a `bio` for the author or `photos` of them

# Looking at some specific examples

As a complement to the high level statistics it's useful to look at some specific example texts.
I picked some technical books I'm aware of and [looked through their records in a notebook](https://github.com/EdwardJRoss/bookfinder/blob/master/notebooks/0031-openlibrary-sql-examples.ipynb).

## Duplicate works

Searching for books with the title "Bayesian Data Analysis" (in a case insensitive way) returned 4 separate works, all clearly the same book.
Note that one book has the author name in the wrong order (`Gelman Andrew`), and that only `/works/OL18391964W` contains all the authors (including Andrew Gelman twice, the second time as `A. Gelman`).

| work_key           | works_title            | author_name     | author_key          |
|--------------------|------------------------|-----------------|---------------------|
| /works/OL25152967W | Bayesian Data Analysis | Gelman Andrew   | /authors/OL9492748A |
| /works/OL12630389W | Bayesian data analysis | Andrew Gelman   | /authors/OL2668098A |
| /works/OL19124056W | Bayesian data analysis | Andrew Gelman   | /authors/OL2668098A |
| /works/OL18391964W | Bayesian data analysis | Andrew Gelman   | /authors/OL2668098A |
| /works/OL18391964W | Bayesian data analysis | John B. Carlin  | /authors/OL2692132A |
| /works/OL18391964W | Bayesian data analysis | Hal S. Stern    | /authors/OL2692133A |
| /works/OL18391964W | Bayesian data analysis | Donald B. Rubin | /authors/OL1194305A |
| /works/OL18391964W | Bayesian data analysis | A. Gelman       | /authors/OL2692134A |


## Duplicate editions

Sometimes an edition is duplicated, such as [How to solve it](https://openlibrary.org/books/OL18335079M/How_to_solve_it) and [How to solve it](https://openlibrary.org/books/OL4468213M/How_to_solve_it) which both have the same pair of ISBN 10 `[0691080976, 0691023565]`

## Multiple works for an edition

Sometimes an edition has multiple works, but all the cases I've checked seem to be errors.


# Summary of Open Libarary

Open Library has massive coverage of books, often with other useful information about the books, but with some duplication and inconsistencies (e.g. among publishers).
It's a good starting point for a knowledge base, but requires additional work to remove duplicates and other errors.
A lot of these are driven by the interface; an interesting extension would be to look more into how the field usage varies by source, and what the sources of duplication are.
These could potentially be improved by the Open Library team to create better results in the future.
But it's still useful enough to work with as is, if we're careful.