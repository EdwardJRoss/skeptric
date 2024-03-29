---
categories:
- python
date: '2021-04-16T21:05:54+10:00'
image: /images/scrapy.png
title: Not Using Scrapy for Web Scraping
---

[Scrapy](https://docs.scrapy.org/en/latest/index.html) is a fast high-level web crawling and web scraping framework.
But as much as I want to like it I find it very constraining and there's a lot of added complexity and magic.
If you don't fit the typical use case it feels like a lot more work and learning doing things with scrapy than without.

I really like [Zyte](https://www.zyte.com/) (formerly ScrapingHub) the team behind Scrapy.
They really know what they're talking about with great blogs about [QA of Data Crawls](https://www.zyte.com/blog/a-practical-guide-to-web-data-qa-part-v-broad-crawls/), [guide to browser tools](https://www.zyte.com/blog/web-scraping-basics-tools/), [how bots are tracked](https://www.zyte.com/blog/how-to-scrape-the-web-without-getting-blocked/), and Scrapy's documentation has a very useful page on [selecting dynamically-loaded content](https://docs.scrapy.org/en/latest/topics/dynamic-content.html).
They've also released a huge number of great open source tools; [parsel](https://github.com/scrapy/parsel) combines the best of lxml and BeautifulSoup for extracting from HTML, [extruct](https://github.com/scrapinghub/extruct) is fantastic for extracting metadata, a [disk based queue library](https://github.com/scrapy/queuelib), an [implementation of simhash](https://github.com/scrapinghub/python-simhash), and flexible parsers for [dates](https://github.com/scrapinghub/dateparser), [prices](https://github.com/scrapinghub/price-parser/), and [numbers](https://github.com/scrapinghub/number-parser/).

But I find Scrapy itself to be a magic monolith; if you try to use it in ways that aren't intended it's quite difficult.
Suppose that you want to extract all the product pages from a large website exactly once, and new products are being added over time.
You get to the product pages through category pages that are updated when new products come in.
It's not at all obvious how to do this in Scrapy.
With the default settings you will rescrape the pages every time (which may not be feasible or economical, or result in some pages being missed).
You can [enable a job queue](https://docs.scrapy.org/en/latest/topics/jobs.html), which as a side effect keeps a list of visited pages that it doesn't revisit.
But you want to revisit the category pages; you can do this by [setting a custom DUPEFILTER_CLASS](https://docs.scrapy.org/en/latest/topics/settings.html#dupefilter-class) or passing `dont_filter` as `True` in the Requests.
However you really only want to revisit the category pages after some time interval; there doesn't seem to be an easy way to do this and maybe you need to manage the state yourself in your spiders using an external database (which could be error prone!).
After spending quite a bit of time looking into this I stumbled across the [DeltaFetch middleware](https://www.zyte.com/blog/scrapy-tips-from-the-pros-july-2016/) which sounds like it can handle this case (by not returning items on the category pages, but on the product pages).

Rather than providing functionality Scrapy seems to lock you into using their tooling.
Extracting data from a webpage is inherently experimental and iterative.
They've got scrapy shell which makes it easier to do some experimentation with IPython, but then you need to copy all your selectors into a file.
Ideally I'd use something like [nbdev](/nbdev) to iterate in a Jupyter notebook and keep the examples and experimental code.
I can hack together examples like below, but I can't work out how to fetch a page with through Scrapy (the equivalent of `scrapy fetch`).

```python
from scrapy.http import TextResponse
import requests

url = 'http://quotes.toscrape.com/page/1'
res = requests.get(url)
response = TextResponse(res.url, body=res.text, encoding='utf-8')

response.css('title::text').get()
```

Scrapy assumes you're wanting to do everything; from getting URLs to parsing and processing at once.
But as I outlined in [web scraping architecture](/select-fetch-extract-watch), I think it's a lot safer to download the data as WARC and then separately extract and normalise the data; moreover this is a great way to write test cases.
If you make a mistake in your extraction code you can raise an error that brings down the scraper.
It's possible to do this in Scrapy, either with a [custom downloader middleware](https://stackoverflow.com/a/27178219), or by directly invoking the `parse` methods from the spiders, but its clunky which makes reprocessing hard.
It sounds like [web-poet](https://web-poet.readthedocs.io/en/stable/) aims to help separate the extraction from I/O, but it's in early stages.

I think there are a lot of cases where Scrapy is a really good choice.
When it doesn't fit it seems much harder to bend it to your will than to cobble together a solution in Python.
Maybe for very large scale projects (or where fast scraping is required) it's worth the investment in Scrapy, but some of the design decisions make it painful for the way I want to use it and for small projects I'm not convinced it's worth the effort.