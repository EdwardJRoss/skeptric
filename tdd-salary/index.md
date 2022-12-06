---
categories:
- python
date: '2020-08-10T08:00:00+10:00'
image: /images/salary_test.png
title: Test Driven Salary Extraction
---

Even when there's a specific field for a price there's a surprising number of ways people write it.
This is what the tool [price-parser](https://github.com/scrapinghub/price-parser) solves.
Unfortunately it doesn't work too well on salaries, which tend to be ranges and much higher, but the approach works.

Price parser has a very large set of tests covering different ways people write prices.
The solution is a simple process involving a basic regular expression, but it solves all these different cases.

When extracting the value and period of salaries from [job ads](/common-crawl-job-ads) I took the same approach.
I looked through a bunch of example data to find different patterns, and especially things that could go wrong.
This gave me over fifty test cases.

It was very easy to write these test cases and use them to work out how to write a parser.
I started with the approach from price parser and modified it to deal with things in salaries, such as ranges like \$50-70k.

Regular expressions are finicky and a test based approach let me quickly see if I broke anything when I changed the expression (as I often did).
However within an hour I could get all but a couple of very hard (and obscure) tests passing, which I removed.

For this problem starting with the tests was the best way forward.
As I discover new issues I can add tests and make sure any extensions or changes don't break existing examples.