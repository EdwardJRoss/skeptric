---
categories:
- hnbooks
date: '2022-08-01T22:04:49+10:00'
image: /images/hn_isbn_open_library.png
title: Open Library ISBN Lookup
---

I've been looking at [Open Library](/open-library) as a knowledge base for books.
After [extracting ASINs from Hacker News posts](/hn-asin), I want to link them to Open Library records.

For books an ASIN is the same as its ISBN-10, which creates a linkage point with Open Library.
From my previous [Open Library Exploration](/openlibrary-exploration) about 20% of editions in Open Library has an ISBN-13, but not an ISBN-10 (and 64% have one of either).
It's [straight-forward to convert](https://bisg.org/page/conversionscalculat/Conversion--Calculations-.htm) an ISBN-10 to an ISBN-13; just prepend "978" and change the final check digit.
Here's how this can be done in Python (you can see the rest of the details in the [notebook](https://github.com/EdwardJRoss/bookfinder/blob/master/notebooks/0033-open-library-isbn-lookup.ipynb)).


```python
isbn_13_weighting = [1,3,1,3,1,3,1,3,1,3,1,3,1]

def isbn13_check_digit(isbn13: str) -> str:
    assert len(isbn13) == 13
    digits = [int(x) for x in isbn13[:-1]]
    check = 10 - sum(x*y for x,y in zip(digits, isbn_13_weighting)) % 10
    
    if check == 10:
        check_digit = "0"
    else:
        check_digit = str(check)
        
    assert len(check_digit) == 1
    assert check_digit in ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    return check_digit
    
def isbn10_to_13(isbn10: str) -> str:
    return "978" + isbn10[:-1] + isbn13_check_digit("978" + isbn10)
```

Trying to match each extracted ASIN with an Open Library record gives 94% matches; 20% with more than one match.
Some of the missing records are in Open Library, just not under this ISBN (for example [Physics for Entertainment](https://openlibrary.org/works/OL8643229W/Physics_for_Entertainment) isn't under 1610279034), and others are missing completely like [Complexity Economics](https://www.amazon.com/dp/1947864351).

The duplicate matches and high coverage align with my earlier investigations with [adding a book to Open Library](/adding-open-library) and looking up books in [Open Library with SQLite](/openlibrary-sqlite).
The books are often imported from multiple places and easy to add multiple times.
For duplicate records we would ideally have some methods for merging their information into a single record.

Now that I can get from a Hacker News post to an Open Library record the next step is to put it all together.
A simple way to do this would be to create a HTML page for every work containing information from Open Library (such as the title and authors), along with all the related Hacker News comments.
This would be enough to prove out the concept before doing additional work to link books described in other ways to Open Library.