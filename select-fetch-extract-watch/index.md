---
categories:
- data
date: '2021-04-15T07:55:57+10:00'
image: /images/select_fetch_extract_watch.png
title: 'Select, Fetch, Extract, Watch: Web Scraping Architecture'
---

The internet is full of useful information just waiting to be collected, processes and analysed.
While getting started scraping data from the web is straightforward, it's easy to tangle the whole process together in a way that makes it fragile to failure, or hard to change with requirements.
And the internet is inconsistent and changing; in anything but the smallest scraping projects you're going to run into failures.
I find it useful to conceptually break web scraping down into four steps; selecting the data to retrieve, fetching the data from the internet, extracting structured data from the raw responses, and watching the process to make sure it's functioning correctly.

When you're getting started put all your work into making sure you can get the information you need.
But before you scale it up operationally think about trying to disentangle the four steps to make it more robust and flexible.


The rest of this article is going to go into detail on each of the four steps, through the lens of three types of use cases:

1. monitoring a single page for changes over time like [VisualPing](https://visualping.io/)
2. extracting information from all the product pages on a website
3. collecting a wide crawl of the web like [Common Crawl](https://commoncrawl.org)

# Select

Select is about choosing the data to extract, where to get it from, and when to retrieve it.

For example if you're monitoring a single page for changes over time then you just need to keep track of a single URL to retrieve.
The only choice you need to make is how often you retrieve it; this could be handled with a stateless scheduler (e.g. run on the first minute of every hour) or by tracking when it was last retrieved (e.g. if it was more than an hour ago then refetch).

If you're extracting all the product pages on a website you'll need a way to find a list of all those pages.
The best way to do this will depend on the website; maybe there's a sitemap.xml with all the pages listed, if they can be retrieved from a sequential numeric id you could search the space of allowed ids, but otherwise you may have to scrape category pages or search results to find all the product page URLs.
The order doesn't normally matter too much if you can scrape all the required pages in the allotted time.
You also need to decide how often to retrieve the information; if it's unlikely to change you may just want to fetch it once, but you may want to check back every month to see what's changed.
Generally you'll get more product pages over time so it's important to keep track of what you've already retrieved and when you retrieved it.

For collecting a wide crawl of the web you start with a small set of seed pages and collect all the URLs in already scraped pages as future sites to crawl.
Here there's too much to scrape every possible page, and you have to decide how to prioritise the crawl (going deep in a few websites, covering as many websites as possible) and when to retrieve a page again.
You will also want to limit the rate at which you request from any given domain in the fetch step.

Select is the most robust phase to failure; the most common reasons it would fail is if a page is removed (for example because a product is no longer offered, or a dead link in a crawl), or more rarely because the URL structure is completely changed.

# Fetch

Fetch is about actually retrieving the data over the network.
There can be all sorts of issues in doing this, and sometimes you only get one shot - especially if you're monitoring changes over time.
Because of this I recommend serialising everything in the Web Archive format so you keep your data; this is easy to do with [Python's requests library](/request-warc) or with [Scrapy](https://github.com/odie5533/WarcMiddleware).

The issues here are fairly common across use cases, and well written about elsewhere.
You need to make sure you're being respectful; understanding the robots.txt, setting an appropriate User Agent, making requests at a rate that adds little load to the server and slowing the rate when there are server issues or the response time increases.
Sometimes you need work around issues, like retrying failed requests, managing sessions and even using proxies.
Overall you want to hit the remote server as few times as possible, so save all your data as WARC so you only need to do it once.

If you've got a huge scraping project keeping all the bulk of HTML and headers in WARC for the few fields you're extracting (often 10x-100x larger after compression) may get burdensome.
But I think of it as a backup; you can delete old ones over time as you're more sure you don't need them, but it's always worth keeping so you can recover from issues in extraction.

# Extract

Extracting is getting the data out of your responses.
If you're doing a wide crawl then the WARC is probably as much extraction as you can do, if you're hitting a well structured API the extraction may already be done in the JSON response.
But it's very common to have to parse data out of HTML, or at least transform some JSON.

For monitoring part of a single page this could be a CSS selector that selects the content to be monitored, and potentially normalising irrelevant changes (e.g. whitespace in the markup).
This could fail if the structure of the page changes so much the section disappears.

A product page is more complicated and the best way to extract the required fields depends on how the page is structured.
But it's a good bet that your heuristics will eventually fail on unusual pages, or as the website changes its structure over time.
A more subtle failure mode is you may discover some pages have a field that would add a lot of value that you weren't previously extracting.
For these reasons it makes sense to work off of WARC files you can retry on later, and skip over errors to extract the data that you can.
Then we can watch these issues and improve the extraction iteratively.

# Watch

Watch is monitoring your pipeline to make sure when it's broken you know, before you end up with weeks of lost data.

For visiting product pages or a web crawl you could watch the select step to make sure you're getting, and fetching, new items at an expected rate.

For all use cases it's useful monitoring the response status codes; if you're getting too many bad codes something is going wrong and you want to raise an alert.
It's also worth doing some sanity checks on the data; the response size should be variable within a certain range (a common failure case is getting empty or error pages).

For extract you can monitor the percentage of fetches that don't extract cleanly and pass validations; they can alert you to a change in the page structure, or even errors with fetching.

# Putting them together

In reality you can't always disentangle these steps.
For a web crawl your selection will depend on the website you just extracted.
Some websites will force you to perform a series of actions to fetch that may impact your choice of selection.
For product information you may need to tie information from a category or search page with data from a product page.

But in the whole the cleaner the interface you can have between the steps, the easier time you'll have maintaining your scraper and recovering from errors.
Notably the [Scrapy architecture](https://docs.scrapy.org/en/latest/topics/architecture.html) follows a similar pattern; the scheduler selects, the downloader fetches, and the spiders extract into the pipeline.