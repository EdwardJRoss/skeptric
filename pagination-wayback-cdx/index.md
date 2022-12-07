---
categories:
- data
date: '2021-11-23T08:00:00+11:00'
image: /images/wayback_empty_returns.png
title: Pagination in Internet Archive's Wayback Machine with CDX
---

I've been trying to use pagination with the [Internet Archive's CDX](https://github.com/internetarchive/wayback/tree/master/wayback-cdx-server) for the [Wayback Machine](https://archive.org/web/) but have been getting *lots* of empty results.
The reason is that filters are applied *after* pagination, and so when using tight filters almost all records will be empty.

## Using the CDX API

Following the documentation it's easy enough to make a query (and similar, but not the same, as [Common Crawl's CDX Index](/searching-100b-pages-cdx)).
Here's some logic to get captures from en.wikipedia.org made on 2021-11-01.

```python
IA_CDX_URL = 'http://web.archive.org/cdx/search/cdx'
DEFAULT_PARAMS = {
                     'url':'https://en.wikipedia.org/*', 
                     'from':'20211101', 
                     'to':'20211101', 
                     'output':'json',
}
def cdx_request(**params):
    return requests.get(IA_CDX_URL, params={**DEFAULT_PARAMS, **params})
```

The returned data is a JSON array of arrays; the first array being the header and the following being the data.

```
'[["urlkey","timestamp","original","mimetype","statuscode","digest","length"],
["org,wikipedia,en)/", "20211101002442", "http://en.wikipedia.org/", "warc/revisit", "-", "3I42H3S6NNFQ2MSVX7XZKYAYSCX5QBYJ", "615"],
["org,wikipedia,en)/", "20211101002443", "https://en.wikipedia.org/", "warc/revisit", "-", "3I42H3S6NNFQ2MSVX7XZKYAYSCX5QBYJ", "935"],
["org,wikipedia,en)/", "20211101022822", "http://en.wikipedia.org/", "unk", "301", "3I42H3S6NNFQ2MSVX7XZKYAYSCX5QBYJ", "740"],
["org,wikipedia,en)/", "20211101022822", "https://en.wikipedia.org/", "text/html", "301", "3I42H3S6NNFQ2MSVX7XZKYAYSCX5QBYJ", "1066"],
...
```

In this case we've got 1574 rows plus the header.

## Adding Pagination

According to the documentation we can also use pagination (which uses a different index).

> To determine the number of pages, add the showNumPages=true param. This is a special query that will return a single number indicating the number of pages

And if we call that:

```python
num_pages_request = cdx_request(showNumPages=True)
num_pages_request.content
# b'11326\n'
```

This seems like way too many pages given there are only a few thousand results, and if we start looking at the pages they seem empty:

```python
page_0_request = cdx_request(page=0)
page_0_request.content
# b'[]\n'
```

The reason is that the pagination is done before the date filter; if we change the dates we get the same result (note that we can't use `from` as a named argument because it's a keyword):

```python
num_pages_request = cdx_request(**{"from":'201801', "showNumPages":"true"})
num_pages_request.content
# b'11326\n'
```

So pagination is only really useful if you want to get almost all the results.
In this case looking for data from a particular day is like finding a needle in the haystack; I suspect there is data in one of the 11326 pages but it will take a long time to find the right one.

In general if you only want Wayback Machine snapshots from a limited period time with CDX you're better off ignoring the pagination entirely and just getting all the results in one query (as long as it's under the internal limit of 150,000).