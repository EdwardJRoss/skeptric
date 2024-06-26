---
categories:
- python
- linux
date: '2021-04-20T08:00:00+10:00'
image: /images/parsel_cli.png
title: A Command Line Interface for HTML With parsel-cli
---

There are many great command line tools for searching and manipulating text (like `grep`), columnar data (like `awk`), JSON data (like `jq`).
With HTML there's [`parsel-cli`](https://github.com/rmax/parsel-cli) built on top of the wonderful [`parsel`](https://parsel.readthedocs.io/en/latest/) Python library.

Parsel is a fantastic library that gives a simple and powerful interface for extracting data from HTML documents using CSS selectors, Xpath and regular expressions.
Parsel-cli is a very small utility that lets you use parsel from the command line (and can be installed with `pip install parsel-cli`).

For example if you wanted to extract all links from a HTML document; you could use `parsel-cli 'a::attr(href)'`
You could also use it to extract particular useful data from a website without an API; for example to get the headlines from Hacker News you can use `curl -q https://news.ycombinator.com/ | parsel-cli '.storylink::text'`
While it's limited compared to actually writing scripts with parsel (especially only being able to extract one field), it's a useful companion for [transforming data in shell](/shell-etl).