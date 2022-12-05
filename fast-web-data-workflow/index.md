---
categories:
- data
date: '2021-11-21T13:49:13+11:00'
image: /images/web_processing_log.png
title: Fast Web Dataset Extraction Worfklow
---

I'm currently streamlining the process of building a dataset from web data.
I want to make it easy for anyone to build their own dataset in a few hours, which requires making the process as smooth as possible.
One thing that makes a major difference is caching intermediate results; it makes the process much faster when you hit an error.
But done naively it can be confusing; if you change a function you want the cache to be recalculated.

# Web Extraction Workflow

I'm focused on a "hobby" size workflow of building a dataset for a data science portfolio project on a mid-range laptop.
This could range from one domain to scores of domains, and from a few webpages to hundreds of thousands of pages.
The trade-offs and approach for larger projects may be different.

I find the workflow tends to be iterative; first identify some fields I want to extract from the webpage.
Then interactively explore a few different options before deciding on what looks like a robust solution, trying them across a few different examples.
Next run it in batch, and invariably find that some examples raise an error, or return data that is wrong or invalid (after some analysis).
This typically involves a few cycles of excluding webpages that don't contain the right kind of data, of amending extraction rules, and of having fallback strategies for unusual pages or to handle changes over time.

Even once a process is successfully running you can run into trouble sometime later.
Sometimes an issue becomes apparent after downstream analysis, or there's another field that should be added (especially one that only occurs on some pages), or when running on a newer snapshot the process breaks.
In all these cases you'll have to go back and re-evaluate your extraction, or your normalisation.

# Making it faster

For a good developer experience it's important to have a fast feedback cycle.
When exploring approaches being able to experiment quickly and get results in under about a second helps keep focus and entering a state of flow.
Being able to run the whole pipeline the faster the more cycles we have to find and correct issues in changes.
There are two main ways to make the process faster; not doing things twice, and doing many things at once.

## Not doing things twice

It may be worth caching the input webpages if you can spare the disk space.
Downloading a large number of webpages over a network can be slow so having the pages locally makes things faster (and is cheaper for whoever is providing the data!).
However individual webpages can range from a few kilobytes to hundreds of kilobytes (and that's just the HTML!).
At the upper end with 100,000 pages each weighing 100 KB, the total size of raw files is around 10GB, which is a reasonable amount of space to use on a hard drive.
Typically they compress well (especially pages from the same website) and you could reduce the usage by an order of magnitude by compressing them.
However an advantage of using the uncompressed HTML files is they can be easily viewed in a browser for quick debugging sessions.

It can also be worth caching the extraction steps (which will be much smaller), but with the caveat that when we change the extraction we want to reprocess everything, and if we're being careful to see what changed to ensure we didn't break anything.
It may be worth separating the transformation process into a straightforward extraction step (which can be computationally costly because it involves processing a large HTML file, especially when using something like BeautifulSoup) and one or more involved validation and normalisation steps (which may be faster, but is more likely to break on strange data).

Each extraction step could process all pages at once (making it easy to check changes), or process files by a time range or folder (making it easy to incrementally add and test new data).
I'm not sure what the best approach is here, it makes sense to start with processing things all together and then adding an ability to process certain batches if that's getting too slow.

## Doing many things at once

The process of extracting data from many web pages is trivially paralellisable; the data can be extracted independently of each other by pure functions.
On a laptop the easiest way to take advantage of this is to run multiple threads or processes to make sure all the CPU cores are being used.
In Python this can be done using `mulitprocessing` spawning one process per CPU core.

In particular downloading data is best done in parallel; I've found having multiple threads for downloading Common Crawl data seems to improve the speed almost linearly up to around 30.
I suppose since the data are typically living on separate servers they can be fetched concurrently, reducing the overhead of waiting for a request to be completed.

In fact the whole process of extracting, transforming, and aggregating the data fits nicely into a simple map reduce framework.
If you wanted to exercise some simple Data Engineering skills it could be interesting to distribute the computation over a cluster (using something like [Ray](https://www.ray.io/), [Dask Distributed](https://distributed.dask.org/en/latest/), [PySpark](http://spark.apache.org/docs/latest/api/python/) or even something like [pywren](http://pywren.io/) or [GNU Parallel](https://www.gnu.org/software/parallel/)).
However it certainly can run on a commodity laptop with no configuration, it may just take hours on large pipelines.

# Dealing with Failure

Web data extraction is a noisy process and you're not always bothered in getting *every* example right.
Generally if you can get it to extract the right data most of the time you're happy, and so you don't want your engine to stop on every failure to extract the right fields.
However you do want to *know* what's failed, so it makes sense to log this, ideally in a structured format for analysis.

A simple way to do this is to run every (pure) extraction function in a try-catch, and record any failures to a log rather than catching them.
Ideally it would be easy to identify different types of errors, so this log should capture identifying information (especially if it comes from validating a field).
An added benefit is you still have a working (although perhaps even more biased) dataset even when there are still failures.

Whenever a failure is resolved it can be added as a test case (which may involve references to specific web pages) to avoid regressions.

# Separation of Concerns

It would be nice when writing an extraction function not to have to worry about all this.
The extraction functions could be pure data processing, and then be executed by an engine that handles caching, parallelism and failures.
This would make it easy to switch out the engine as needs change without touching all the data extraction functions.

However when debugging it could be handy to have some hooks back into cached data to make the process faster.
There seems to be a genuine trade-off here and it will require some experimentation to find the right balance.

# Implementation

There are many ways to approach the caching and parallelism.
My first thought would be something like [shelve](https://docs.python.org/3/library/shelve.html) or [sqldict](https://github.com/RaRe-Technologies/sqlitedict) (or plain old pickle) for caching data, and multiprocessing for paralellism.
However [Joblib](https://joblib.readthedocs.io/en/latest/) looks like a good solution that handles both of them in an elegant way that may be good enough without much extra coding.
What I need to work out is how much transparency is needed in the workflow - inevitably users will run into problems and the easier it is for them to see how these processes work the quicker they can fix them.