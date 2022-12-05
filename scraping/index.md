---
date: 2019-11-10 22:20:40+11:00
draft: true
title: Tips for data scraping
---

The internet is one of the best data sources with readily available interesting datasets.
Scraping is the art of extracting that data in a useful form.

# 0. Scope out project

What data do you need?
How much?

# 1. Look for resources

## Dumps
https://commoncrawl.org/

https://console.cloud.google.com/marketplace/details/y-combinator/hacker-news

https://console.cloud.google.com/marketplace/details/github/github-repos

https://archive.org/details/twitterstream

http://files.pushshift.io/reddit/

https://en.wikipedia.org/wiki/Wikipedia:Database_download

https://archive.org/details/stackexchange

Amazon Reviews: http://academictorrents.com/details/66ddbb6d5f49aa6c36a01ca5e814f1beef00b5b7

## APIs
https://developer.github.com/v3/
https://developer.twitter.com/en/docs
https://www.reddit.com/dev/api/

## Understand legal obligations

TOS
robots.txt

https://blog.scrapinghub.com/solution-architecture-part-3-conducting-a-web-scraping-legal-review

More conservative:
https://benbernardblog.com/web-scraping-and-crawling-are-perfectly-legal-right/

# 2. Learn how to navigate

* Gathering a list

* Iterating through sequences (ignoring structure)

* sitemap.xml

* Walking tree structures (crawling)

* Walking Search Listings

* Infinite search, JS and XHR Requests



# 3. Learn how to get the data


View source and XHR

* Javascript in the browser
e.g. google image search fastai

* API endpoints

* JSON blobs in HTML

* BeautifulSoup/XPath


# 4. Troubleshooting

* On logins

* On javascript

* On useragents and proxies


# 4. Schedule it

* Rate limiting

* State, etc.

* Monitoring