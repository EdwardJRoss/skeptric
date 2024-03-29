---
categories:
- commoncrawl
- data
date: '2020-06-12T08:00:00+10:00'
image: /images/warcinfo.png
title: Extracing Text, Metadata and Data from Common Crawl
---

[Common Crawl](https://commoncrawl.org) builds an open dataset containing [over 100 billion unique items](https://commoncrawl.github.io/cc-crawl-statistics/plots/crawlsize) downloaded from the internet.
[You can search the index](/searching-100b-pages-cdx) to find where pages from a particular website are archived, but you still need a way to access the data.
Common Crawl provides the data in 3 formats:

* If you just need the text of the internet use the WET files
* If you just need the response metadata, HTML head information or links in the webpage use the WAT files
* If you need the whole HTML (with all the metadata) then use the full WARC files 

The index only contains locations for the WARC files, the WET and WAT files are just summarisations of it.
For more detail read [Common Crawl's introduction](https://commoncrawl.org/2014/04/navigating-the-warc-file-format/) and the [WARC specification](https://iipc.github.io/warc-specifications/specifications/warc-format/warc-1.0/).
For processing these at scale see [Common Crawl's pyspark code samples](https://github.com/commoncrawl/cc-pyspark) and [Mark Litwintschik's post on using EMR and Spark with Common Crawl](https://tech.marksblogg.com/petabytes-of-website-data-spark-emr.html).

See the [Jupyter Notebook](/notebooks/WAT%20WET%20WARC%20-%20Common%20Crawl%20Archives.html) ([Raw](https://nbviewer.org/github/EdwardJRoss/skeptric/blob/master/static/notebooks/WAT%20WET%20WARC%20-%20Common%20Crawl%20Archives.ipynb)) for more code samples.

# A brief introduction to WARC record types

Each of these files (WARC, WET, and WAT) are stored in WARC format.
The specification lists a few possible types of records which are useful to know about: 

* warcinfo - contains information about the web crawl (normally the first record)
* request - details of HTTP request
* response - details from HTTP response
* metadata - additional information
* conversion - result of transforming another field
* resource - record contains a resource (e.g. image)

There's also a couple odd kinds I won't talk about, but may come up.

* revisit - describes the revisitation of content already archived, may only contain changed definition
* continuation - appended to corresponding prior record block(s) (e.g., from other WARC files) to create complete record

# Just the Text - WET

The WET contains just the webpage title, and plain text extracted from the HTML of each response.
This could be useful for text analysis, building a search index or [training a machine learning language model](https://github.com/facebookresearch/cc_net).
It's about 1/6 the size of the full WARC files.

All the archives can be accessed with the [warcio library](https://github.com/webrecorder/warcio) using ArchiveIterator.
You can access the WET file by changing the URL to a WARC file.

```python
from warcio import ArchiveIterator
wet_url = warc_url.replace('/warc/', '/wet/').replace('warc.gz', 'warc.wet.gz')
r = requests.get(warc_url, stream=True)
records = ArchiveIterator(r.raw)
```

The first record is a `warcinfo` record describing the crawl, and all the following requests are `conversion` records containing the plain text of each response.

```python
record = next(records)
assert record.rec_type == 'warcinfo'
# skip the warcinfo
record = next(records)
# This shows the source page, WARC-Target-URI and other metadata
record.rec_headers.headers
text = record.content_stream().read()
print(text.decode('utf-8'))
```

# Metadata - WAT

The WAT contains just the metadata from each page, the request info, things from head of HTML, and links from the webpage.
You could use this for understanding what web servers most commonly used (from response headers), for analysing declared keywords or for analysing the link structure (finding reverse links or calculating [page rank](https://en.wikipedia.org/wiki/PageRank)).
It also contains details of the corresponding WARC record so you could use the WAT data to find the WARC files you need before extracting the full HTML from them.
It's about 1/3 the size of the full WARC files because it doesn't contain the actual content.

You can access the WAT file by changing the URL to a WARC file:

```python
wat_url = warc_url.replace('/warc/', '/wat/').replace('warc.gz', 'warc.wat.gz')
```

The first record is a `warcinfo` describing the crawl, followed by `metadata` records with JSON content.
It seems that the first `metadata` describes the corresponding full WARC file, and then the following `metadata` records are aligned to corresponding records in the `WARC` file.

```python
record = next(records)
assert record.rec_type == 'warcinfo'
# skip the warcinfo
record = next(records)
# Headers tell us what the record is about (e.g. source url)
record.rec_headers.headers
metadata = json.loads(record.content_stream().read())
```

The metadata JSON object has a `Container` key describing the corresponding WARC source, and an `Envelope` describing the record itself.
If it's describing a `response` (rather than a `request` or `metadata` or something else) you can access the `HTML-Metadata`.

```python
data['Envelope']\
    ['Payload-Metadata']\
    ['HTTP-Response-Metadata']\
    ['HTML-Metadata']
```

Here's some example content:

```
{'Head': {'Title': '纺织服装行业周报:终端零售回暖,板块业绩等待验证 - 相关研报 - 梦洁股份(002397)',
  'Metas': [{'name': 'mobile-agent',
    'content': 'format=html5; url=detail_m.php?id=866619'},
   {'name': 'mobile-agent',
    'content': 'format=xhtml; url=detail_m.php?id=866619'},
   {'name': 'keywords',
    'content': '纺织服装行业周报:终端零售回暖,板块业绩等待验证,相关研报,梦洁股份,002397'},
   {'name': 'description',
    'content': '梦洁股份(002397)相关研报：纺织服装行业周报:终端零售回暖,板块业绩等待验证'}],
  'Link': [{'path': 'LINK@/href',
    'url': 'http://txt.inv.org.cn/ir/site/pc/css.css',
    'rel': 'stylesheet',
    'type': 'text/css'}],
  'Scripts': [{'path': 'SCRIPT@/src',
    'url': 'http://static.bshare.cn/b/buttonLite.js#style=-1&uuid=&pophcol=2&lang=zh',
    'type': 'text/javascript'},
   {'path': 'SCRIPT@/src',
    'url': 'http://static.bshare.cn/b/bshareC0.js',
    'type': 'text/javascript'},
   {'path': 'SCRIPT@/src',
    'url': '//pagead2.googlesyndication.com/pagead/js/adsbygoogle.js'},
   {'path': 'SCRIPT@/src',
    'url': '//pagead2.googlesyndication.com/pagead/js/adsbygoogle.js'}]},
 'Links': [{'path': 'A@/href',
   'url': '/',
   'target': '_blank',
   'text': '梦洁股份(002397)'},
  {'path': 'A@/href',
   'url': '/index_m.php',
   'target': '_blank',
   'text': '移动版'},
  {'path': 'IMG@/src', 'url': 'http://img.inv.org.cn/broker/huasheng_pc.jpg'},
  {'path': 'A@/href',
   'url': 'https://hd.hstong.com/marketing/2019/0228?_scnl=OTg0NWJibzY0MTI3',
   'target': '_blank'},
  {'path': 'A@/href', 'url': '/', 'text': '首页'},
  {'path': 'A@/href', 'url': '/quote/', 'text': '股票行情'},
  {'path': 'A@/href', 'url': '/media_news/', 'text': '媒体报道'},
  {'path': 'A@/href', 'url': '/related_news/', 'text': '相关新闻'},
  {'path': 'A@/href', 'url': '/notice/', 'text': '公司公告'},
  {'path': 'A@/href', 'url': '/report/', 'text': '研究报告'},
  {'path': 'A@/href', 'url': '/related_report/', 'text': '相关研报'},
  {'path': 'A@/href', 'url': '/', 'target': '_blank', 'text': '梦洁股份'},
  {'path': 'A@/href', 'url': '/', 'target': '_blank', 'text': '002397'},
  {'path': 'A@/href',
   'url': 'http://www.bShare.cn/',
   'title': '分享到',
   'text': '分享到'},
  {'path': 'IMG@/src', 'url': 'http://img.inv.org.cn/ad/zixun_pc.jpg'},
  {'path': 'A@/href',
   'url': 'http://stock.inv.org.cn',
   'target': '_blank',
   'text': '股票投资之家'}]}
```

# Data - WARC

The WARC files are the ultimate data source.
You really only need to use them if you need to efficiently access the data via an index or you need the actual HTML content.
Sometimes the HTML content is necessary because you want to know about javascript used in the body/foot of the page, or you want the structured content of the page (not just the text).

The WARC files start with a `warcinfo` record describing the file.
This is followed by sequences of records for each event, e.g. accessing a URL.
A typical pattern is a `request` record describing how the content was requested, a `response` record describing what was received from the server including HTTP headers and, and some `metadata` such as detected languages, character sets and the time to fetch the page.

I'm not sure whether there are `resource` or `revisit` records in the WARC (or `continuation` records that sound painful).