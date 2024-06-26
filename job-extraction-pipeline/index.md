---
categories:
- jobs
- python
- commoncrawl
date: '2020-11-09T18:23:31+11:00'
image: /images/job_extraction_pipeline.png
title: Building a Job Extraction Pipeline
---

I've been trying to [extract job ads from Common Crawl](/common-crawl-job-ads).
However I was stuck for some time on how to actually write transforms for all the different data sources.
I've finally come up with an architecture that works; download, extract and normalise.

I need a way to extract the job ads from heterogeneous sources that allows me to extract different kinds of data, such as the title, location and salary.
I got stuck in code for a long time trying to do all this together and getting a bit confused about how to make changes.
So I spent some time thinking through how to do it, and then it was straightforward to implement.

The architecture of download, extract and normalise makes it easy to inspect and incrementally add to the pipeline.
The first step download gets the raw web archives from Common Crawl; this lets me inspect the source HTML directly in a browser of notebook.
The next step extract takes the structured data from the HTML and outputs JSON Lines; this lets my inspect all the structured data I can extract from the webpages for a source.
The final step normalise takes the extracted fields and applies normalisations to get them in a common format; this allows all the data to be aligned and merged.
When adding a new data source I can first use the downloaded data to construct a parser, then extract with the parser to get all the raw data, then finally transform and align it in with a normaliser.
If I want to add a new column to the output I just need to write the corresponding normaliser to every source, as long as the information is in the underlying extracted data.

Initially I was trying to combine parts of the normalise and extract steps, then run the same normalisers to convert [HTML to text](/html-to-text), [extract the salary](/tdd-salary) and [normalise the location](/placeholder-australia).
The problem is that not every source requires the same steps; one source emits plain text rather than HTML, some have the salary amount and period as separate fields rather than together as a string, and some have very different ways of normalising the location.
I was trying to solve all this with the extract step but it got very confusing and it was mixing transformation with extraction.

Writing separate normalisers for each source means I end up with more repeated code, although a lot of it uses common functions.
But it makes it easier to make small adjustments per source, to debug the process and results in a much simpler codebase (simple in the [Rich Hickey sense](https://www.infoq.com/presentations/Simple-Made-Easy)). 
It also allows me to start simply with one or two normalisers such as title and description, and then incrementally add more such as posted date, location, and salary.