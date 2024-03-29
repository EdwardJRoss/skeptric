---
categories:
- nlp
- ner
- hnbooks
date: '2022-06-22T22:12:21+10:00'
image: /images/extracting_asins.png
title: Finding ASINs in HackerNews
---

I'm currently working on a project [to extract books from Hacker News](/book-title-ner-outline).
After exporting all 2021 posts from the Google Bigquery dataset in a [Kaggle Notebook](https://www.kaggle.com/code/edwardjross/hackernews-2021-export/notebook) and doing an [exploratory data analysis](/hackernews-dataset-eda) I'm looking for methods to extract books.

One way to extract books is using Amazon links.
People often refer to a book with a link to Amazon, and each Amazon product has a 10 character ASIN ([Amazon Standard Identification Number](https://en.wikipedia.org/wiki/Amazon_Standard_Identification_Number)); for books this is the same as its ISBN-10.

I spent some time iterating on a regular expression to find these.
I would take a sample of 10,000 posts, look at the top results and check them.
Eventually I settled on `amazon\.[^"> ]*/dp/([A-Z0-9]{10})\W`; this works across different TLDs (e.g. `.com`, `.co.uk`, `.ca`), captures most examples and has few false positives.

There were only under 200 ASINs that have occured more than once in 2021.

| Frequency | Distinct ASINs |
|-----------|----------------|
| 1         | 2243           |
| 2         | 158            |
| 3         | 40             |
| 4         | 7              |
| 5         | 4              |
| 7         | 1              |
| 6         | 1              |

I checked the 10 most common ASINs on Amazon and they were all books.
I also checked another random 10 and most were books, but there were also other products.
Looking through the text it was pretty easy to see which were books and which were electronics.
I started looking for patterns in the text for more ideas on how to extract books and filter out electronics.
But then I realised I was effectively doing annotation work that I didn't capture; instead I exported it to a CSV to enable more analysis and annotation.

If you want to see the gory details check out the [notebook](https://github.com/EdwardJRoss/bookfinder/blob/master/notebooks/0010-extracting-asins.ipynb).