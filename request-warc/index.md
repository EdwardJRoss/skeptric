---
categories:
- python
date: '2020-07-03T08:00:03+10:00'
image: /images/webrecorder.png
title: Saving Requests and Responses in WARC
---

When fetching large amounts of data from the internet a best practice is caching all the data.
While it might seem easy to extract just the information you need, it's easy to hit edge cases or changing structure, and you can never use the data you throw away.
This is easy to do in the Web ARChive (WARC) format with [warcio](https://github.com/webrecorder/warcio) used by the [Internet Archive](https://archive.org/web/) and [Common Crawl](http://commoncrawl.org/).

```python
from warcio.capture_http import capture_http
import requests  # requests must be imported after capture_http


urls = [f'http://quotes.toscrape.com/page/{i}/' for i in range(1, 11)]

with capture_http('quotes.warc.gz'):
    for url in urls:
        requests.get(url)
```

This creates (or appends to an existing) file `quotes.warc.gz` that contains every response and request we sent.
Because it contains all the data we can experiment with processing the data without ever hitting the server again.
The main drawback of keeping all the raw HTML is that it requires an order of magnitude more storage (especially in websites with large reams of javascript and templated code); but often storage is cheap enough that it's worthwhile.
Another thing to consider is data validation should be done almost immediately (or in parallel) to make sure you're not just collecting a large number of 404s or empty pages.

The generated WARC file contains the responses *compressed* within the gzip file, so annoyingly you can't read it with zcat.
However warcio has some command line tools to access it.

We can find what's in the file with `warcio index quotes.warc.gz`; which is each response followed by the request that generated it.

```json
{"offset": "0", "warc-type": "response", "warc-target-uri": "http://quotes.toscrape.com/page/1/"}
{"offset": "2756", "warc-type": "request", "warc-target-uri": "http://quotes.toscrape.com/page/1/"}
{"offset": "3210", "warc-type": "response", "warc-target-uri": "http://quotes.toscrape.com/page/2/"}
{"offset": "6912", "warc-type": "request", "warc-target-uri": "http://quotes.toscrape.com/page/2/"}
{"offset": "7367", "warc-type": "response", "warc-target-uri": "http://quotes.toscrape.com/page/3/"}
{"offset": "10108", "warc-type": "request", "warc-target-uri": "http://quotes.toscrape.com/page/3/"}
{"offset": "10557", "warc-type": "response", "warc-target-uri": "http://quotes.toscrape.com/page/4/"}
{"offset": "13177", "warc-type": "request", "warc-target-uri": "http://quotes.toscrape.com/page/4/"}
{"offset": "13631", "warc-type": "response", "warc-target-uri": "http://quotes.toscrape.com/page/5/"}
{"offset": "16221", "warc-type": "request", "warc-target-uri": "http://quotes.toscrape.com/page/5/"}
{"offset": "16673", "warc-type": "response", "warc-target-uri": "http://quotes.toscrape.com/page/6/"}
{"offset": "19383", "warc-type": "request", "warc-target-uri": "http://quotes.toscrape.com/page/6/"}
{"offset": "19837", "warc-type": "response", "warc-target-uri": "http://quotes.toscrape.com/page/7/"}
{"offset": "22894", "warc-type": "request", "warc-target-uri": "http://quotes.toscrape.com/page/7/"}
{"offset": "23348", "warc-type": "response", "warc-target-uri": "http://quotes.toscrape.com/page/8/"}
{"offset": "26266", "warc-type": "request", "warc-target-uri": "http://quotes.toscrape.com/page/8/"}
{"offset": "26721", "warc-type": "response", "warc-target-uri": "http://quotes.toscrape.com/page/9/"}
{"offset": "29719", "warc-type": "request", "warc-target-uri": "http://quotes.toscrape.com/page/9/"}
{"offset": "30171", "warc-type": "response", "warc-target-uri": "http://quotes.toscrape.com/page/10/"}
{"offset": "32770", "warc-type": "request", "warc-target-uri": "http://quotes.toscrape.com/page/10/"}
```

We can see the headers of the second response, with offset 3210 (from the above index) using `warcio extract --headers quotes.warc.gz 3210`; 
Note that this includes the target URI, the datetime and the status code.

```
WARC/1.0
WARC-IP-Address: 136.243.118.219
WARC-Type: response
WARC-Record-ID: <urn:uuid:5da1f262-2542-4aed-bf3a-768202f1fb7b>
WARC-Target-URI: http://quotes.toscrape.com/page/2/
WARC-Date: 2020-07-02T11:09:00Z
WARC-Payload-Digest: sha1:KSQKJ4N5HKIDDZR5PV7I6PELYGSU3JPD
WARC-Block-Digest: sha1:CVS43VR3MFTMESL74QP347IXEF7Z25YE
Content-Type: application/http; msgtype=response
Content-Length: 3348

HTTP/1.1 200 OK
Server: nginx/1.14.0 (Ubuntu)
Date: Thu, 02 Jul 2020 11:09:00 GMT
Content-Type: text/html; charset=utf-8
Transfer-Encoding: chunked
Connection: keep-alive
X-Upstream: spidyquotes-master_web
Content-Encoding: gzip
```

The actual data can be extracted the sameway with `payload`: `warcio extract --payload quotes.warc.gz 3210 | head -n 20`

```html
<!DOCTYPE html>
<html lang="en">
<head>
        <meta charset="UTF-8">
        <title>Quotes to Scrape</title>
    <link rel="stylesheet" href="/static/bootstrap.min.css">
    <link rel="stylesheet" href="/static/main.css">
</head>
<body>
    <div class="container">
        <div class="row header-box">
            <div class="col-md-8">
                <h1>
                    <a href="/" style="text-decoration: none">Quotes to Scrape</a>
                </h1>
            </div>
            <div class="col-md-4">
                <p>

                    <a href="/login">Login</a>
```

The library is very simple to use and contains a lot of relevant information.
This seems really useful for debugging if things start going wrong later down the pipeline.