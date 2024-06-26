---
categories:
- programming
date: '2020-11-25T20:38:32+11:00'
image: /images/job_advert_file_tree.png
title: Code Structure Reflecting Function
---

I've been trying to [extract job ads from Common Crawl](/common-crawl-job-ads).
However I've been stuck on how to structure the code.
Thinking through the relationships really helped me do this.

The [architecture of the pipeline](/job-extraction-pipeline) is a set of methods that fetch source data, extract the structured data and normalise it into a common form to be combined.
I previously had these methods all written in one large file, adding each extractor to a dictionary, which was a headache to look at.
But whenever I thought about how to restructure it I got stuck thinking about implementation details.

I made a breakthrough today by thinking about how the components interact.
Each fetcher, extractor and normaliser need to come as a bundle; it's useless to have the components separately and so they need to be added and removed as a whole.
On the other hand each source is independent of the others, so they should be in separate files, and common operations should be extracted into library functions.

It's not entirely obvious how to bundle these things and export them to be run.
A fetcher is something that takes an output path and serialises data to that path.
An extractor is something that takes an input path, from the fetcher, and returns an iterable of structured data (the format of which depends on the source).
A normaliser takes an iterable of structured data, from the extractor, and outputs an iterable of Job Data in a common format.
These could be functions at the top level in a module, combined in a dictionary, or stored in an object.

I decided each should contain a single Datasource class that contains the actions as methods, that can be individually imported.
Really an object in Python is just some syntactic sugar around a dictionary, but the sugar of inheritance and `.` access is sweet.
I called each of the classes `Datasource` with the idea that the choices of datasources could become configurable with minimal boilerplate; they're just distinguished by the module name.

I'm still not sure that I've got it *right*, but having the extractors in separate files makes it much more manageable to see what's going on and I can continue to improve the structure as the code evolves.