---
categories:
- jobs
date: '2020-11-21T23:01:28+11:00'
image: /images/job_extraction_sample.png
title: A First Cut of Job Extraction
---

I've finally built a first iteration of a job extraction pipeline in my [job-advert-analysis repository](https://github.com/EdwardJRoss/job-advert-analysis).
There's nothing in there that I haven't written about, but it's simply doing the work to bring it all together.
I'm really happy to have a full pipeline that extracts lots of interesting features to analyse, and is easy to extend.

I've already talked about [how to extract jobs from Common Crawl](/common-crawl-job-ads) and the [architecture for extracting the data](/job-extraction-pipeline).
I'll just briefly summarise how it all fits together.

The data is taken from Common Crawl's scans of the web.
We have a set of predefined URL patterns to search on Common Crawl that correspond to different job ads, that were found through various means.
These are searched for in the [Common Crawl capture index](/searching-100b-pages-cdx) to find where all the corresponding pages are.
Then all the corresponding pages are downlaoded from S3 as WARC.

The pages are then converted to JSON Lines for each source, using a custom extractor for each source capturing the semi-structured data.
These are then normalised (e.g. HTML to text, extract salary, normalise location) and output as a feather dataframe.

With just two crawls (which happen about once a month) I've got over 13,000 job ads.
Now I've got all this together I'm going to start analysing the results; and probably finding some bugs in the pipeline!