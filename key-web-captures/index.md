---
categories:
- data
date: '2021-11-19T12:07:27+11:00'
image: /images/key_web_captures.png
title: Unique Key for Web Captures
---

I'm currently developing a workflow for extracting data from captures of web pages, leveraging large archives like Common Crawl and the Wayback Machine.
A pain point in extracting data from external sources is you have to reverse engineer the schema, and quite often it breaks on certain data.
To reduce the pain of hitting these errors it makes sense to *cache* the results to reduce the time of re-extracting the data; especially if it's doing something expensive like parsing a large HTML document.
However what's the best way to key an individual capture?

Each capture represents a snapshot of a URL at a certain point in time; in this way we could represent a capture with a timestamp and URL.
Another view is each capture is stored somewhere where we retrieve it, and so could be represented by a pointer to its location.
Finally we could just think of the capture by its contents (the HTML page) and represent it by some hash.
These different choices all have different trade-offs summarised in the table below:

|                     | Timestamp/URL | Pointer   | Content Hash |
|---------------------|---------------|-----------|--------------|
| Human Interpretable | Yes           | No        | No           |
| Representation Size | Very Long     | Long      | Short        |
| Always Exist        | No            | Yes       | Yes          |
| Permenant           | Yes           | Typically | Yes          |
| Unique              | Typically     | Yes       | Vulnerable   |


A timestamp/URL is easy to interpret as a person, whereas the pointer and the hash often are opaque as to what the resource is.
However a URL can be very long (especially when it contains a query string with lots of parameters), whereas a pointer has a typical fixed length and a content hash is guaranteed to have a fixed length.
We are guaranteed to have a pointer to the data (otherwise we can't obtain it), and we can always hash the content (although it may be expensive to do so), but there may be cases where we have data but not the URL or time of capture.
The pointer can change if the assets are moved; this can be incorporated by some sort of mapping from old to new locations (assuming it doesn't happen vary often), wheras the Timestamp/URL and Content Hash should never change (assuming the hash algorithm stays the same).

Uniqueness is a bit more subtle; are two captures of the same URL at different times with the same content different?
For the purposes of extraction they are the same - they contain the same data so we will extract the same results.
However we may want to distinguish them if we are monitoring the changes to a web page, so in that sense the timestamp/URL and pointer contain additional information.
We may also potentially capture the same URL twice at the same timestamp, but that's unlikely to happen much and when it does the content is likely to be the same (but not guaranteed, especially if the requests have different parameters and come from different IP addresses).
Finally for a hash of the content there's always the chance of a hash-collision; for example both Common Crawl and the Wayback Machine contain a SHA-1 for which there are [practical algorithms to generate collisions](https://security.googleblog.com/2017/02/announcing-first-sha1-collision.html).
However a random collision is very unlikely; from the [Birthday Problem](https://en.wikipedia.org/wiki/Birthday_problem) the probability of a random collision between n items in a space with m values is approximately $$\frac{n^2}{2m}$$, given that for SHA-1 the space is $$m = 2^{160}$$ we only need to start really worrying about collisions around $$2^{80} \approx 10^{24}$$ distinct documents.

The content hash seems like the best option for a cache key; attacks and collisions are possible but unlikely (and can be mitigated by a change of hash function), but it always exists and has a short representation (which can even be used for a filename on modern filesystems).
Given that WARC currently captures SHA-1 digest, and CDX servers provide it, makes it practical to use (although if they change the algorithm it may lead to some difficulties).
It also makes sense from a functional perspective; if the content is equivalent then a pure function should give always give the same result.
If we want to track other kinds of identity, such as the timestamp and URL of capture, we can store that separately in a database (where the hash itself is not a key).