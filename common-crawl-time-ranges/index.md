---
categories:
- commoncrawl
date: '2021-11-27T21:20:40+11:00'
image: /images/common_crawl_time_ranges.png
title: Common Crawl Time Ranges
---

[Common Crawl](https://commoncrawl.org) provides a huge open web dataset of going back to around 2009.
Unfortunately it's not easy to find out the time period covered in each index, and so I ran a quick job to get rough estimates of the periods.
This is useful if you want to get data from a specific time range.

# Methodology

The best way to do this would be to use the [Athena Columnar Index](/common-crawl-index-athena) to search the dates, but I didn't have Athena set up and I'd have to be a *little* careful about the costs.
As a proxy I used the [CDX Server](/searching-100b-pages-cdx) and for each index searched 'en.wikipedia.org/*' and got the oldest and newest timestamps across the first and last page of results.
This won't exactly get the whole range, but because Wikipedia is frequently crawled this would be a reasonable proxy.

Here are the results showing the earliest and latest days found, as well as the ISO Week Start (so e.g. 2021-43 starts on 2021-10-31).
Most crawls span a couple of weeks or less.

| Index Name        | Crawl Days | Earliest Date | Latest Date | ISO Week Start |
|-------------------|------------|---------------|-------------|----------------|
| CC-MAIN-2021-43   | 13         | 2021-10-15    | 2021-10-28  | 2021-10-31     |
| CC-MAIN-2021-39   | 13         | 2021-09-16    | 2021-09-29  | 2021-10-03     |
| CC-MAIN-2021-31   | 14         | 2021-07-23    | 2021-08-06  | 2021-08-08     |
| CC-MAIN-2021-25   | 13         | 2021-06-12    | 2021-06-25  | 2021-06-27     |
| CC-MAIN-2021-21   | 14         | 2021-05-05    | 2021-05-19  | 2021-05-30     |
| CC-MAIN-2021-17   | 13         | 2021-04-10    | 2021-04-23  | 2021-05-02     |
| CC-MAIN-2021-10   | 13         | 2021-02-24    | 2021-03-09  | 2021-03-14     |
| CC-MAIN-2021-04   | 13         | 2021-01-15    | 2021-01-28  | 2021-01-31     |
| CC-MAIN-2020-50   | 13         | 2020-11-23    | 2020-12-06  | 2020-12-20     |
| CC-MAIN-2020-45   | 13         | 2020-10-19    | 2020-11-01  | 2020-11-15     |
| CC-MAIN-2020-40   | 14         | 2020-09-18    | 2020-10-02  | 2020-10-11     |
| CC-MAIN-2020-34   | 12         | 2020-08-03    | 2020-08-15  | 2020-08-30     |
| CC-MAIN-2020-29   | 14         | 2020-07-02    | 2020-07-16  | 2020-07-26     |
| CC-MAIN-2020-24   | 14         | 2020-05-24    | 2020-06-07  | 2020-06-21     |
| CC-MAIN-2020-16   | 13         | 2020-03-28    | 2020-04-10  | 2020-04-26     |
| CC-MAIN-2020-10   | 13         | 2020-02-16    | 2020-02-29  | 2020-03-15     |
| CC-MAIN-2020-05   | 12         | 2020-01-17    | 2020-01-29  | 2020-02-09     |
| CC-MAIN-2019-51   | 11         | 2019-12-05    | 2019-12-16  | 2019-12-29     |
| CC-MAIN-2019-47   | 12         | 2019-11-11    | 2019-11-23  | 2019-12-01     |
| CC-MAIN-2019-43   | 11         | 2019-10-13    | 2019-10-24  | 2019-11-03     |
| CC-MAIN-2019-39   | 9          | 2019-09-15    | 2019-09-24  | 2019-10-06     |
| CC-MAIN-2019-35   | 9          | 2019-08-17    | 2019-08-26  | 2019-09-08     |
| CC-MAIN-2019-30   | 9          | 2019-07-15    | 2019-07-24  | 2019-08-04     |
| CC-MAIN-2019-26   | 12         | 2019-06-15    | 2019-06-27  | 2019-07-07     |
| CC-MAIN-2019-22   | 8          | 2019-05-19    | 2019-05-27  | 2019-06-09     |
| CC-MAIN-2019-18   | 8          | 2019-04-18    | 2019-04-26  | 2019-05-12     |
| CC-MAIN-2019-13   | 9          | 2019-03-18    | 2019-03-27  | 2019-04-07     |
| CC-MAIN-2019-09   | 9          | 2019-02-15    | 2019-02-24  | 2019-03-10     |
| CC-MAIN-2019-04   | 9          | 2019-01-15    | 2019-01-24  | 2019-02-03     |
| CC-MAIN-2018-51   | 10         | 2018-12-09    | 2018-12-19  | 2018-12-23     |
| CC-MAIN-2018-47   | 10         | 2018-11-12    | 2018-11-22  | 2018-11-25     |
| CC-MAIN-2018-43   | 9          | 2018-10-15    | 2018-10-24  | 2018-10-28     |
| CC-MAIN-2018-39   | 8          | 2018-09-18    | 2018-09-26  | 2018-09-30     |
| CC-MAIN-2018-34   | 8          | 2018-08-14    | 2018-08-22  | 2018-08-26     |
| CC-MAIN-2018-30   | 8          | 2018-07-15    | 2018-07-23  | 2018-07-29     |
| CC-MAIN-2018-26   | 8          | 2018-06-17    | 2018-06-25  | 2018-07-01     |
| CC-MAIN-2018-22   | 8          | 2018-05-20    | 2018-05-28  | 2018-06-03     |
| CC-MAIN-2018-17   | 8          | 2018-04-19    | 2018-04-27  | 2018-04-29     |
| CC-MAIN-2018-13   | 8          | 2018-03-17    | 2018-03-25  | 2018-04-01     |
| CC-MAIN-2018-09   | 9          | 2018-02-17    | 2018-02-26  | 2018-03-04     |
| CC-MAIN-2018-05   | 8          | 2018-01-16    | 2018-01-24  | 2018-02-04     |
| CC-MAIN-2017-51   | 9          | 2017-12-10    | 2017-12-19  | 2017-12-24     |
| CC-MAIN-2017-47   | 8          | 2017-11-17    | 2017-11-25  | 2017-11-26     |
| CC-MAIN-2017-43   | 8          | 2017-10-16    | 2017-10-24  | 2017-10-29     |
| CC-MAIN-2017-39   | 7          | 2017-09-19    | 2017-09-26  | 2017-10-01     |
| CC-MAIN-2017-34   | 8          | 2017-08-16    | 2017-08-24  | 2017-08-27     |
| CC-MAIN-2017-30   | 9          | 2017-07-20    | 2017-07-29  | 2017-07-30     |
| CC-MAIN-2017-26   | 7          | 2017-06-22    | 2017-06-29  | 2017-07-02     |
| CC-MAIN-2017-22   | 8          | 2017-05-22    | 2017-05-30  | 2017-06-04     |
| CC-MAIN-2017-17   | 8          | 2017-04-23    | 2017-05-01  | 2017-04-30     |
| CC-MAIN-2017-13   | 9          | 2017-03-22    | 2017-03-31  | 2017-04-02     |
| CC-MAIN-2017-09   | 10         | 2017-02-19    | 2017-03-01  | 2017-03-05     |
| CC-MAIN-2017-04   | 9          | 2017-01-16    | 2017-01-25  | 2017-01-29     |
| CC-MAIN-2016-50   | 9          | 2016-12-02    | 2016-12-11  | 2016-12-18     |
| CC-MAIN-2016-44   | 9          | 2016-10-20    | 2016-10-29  | 2016-11-06     |
| CC-MAIN-2016-40   | 8          | 2016-09-24    | 2016-10-02  | 2016-10-09     |
| CC-MAIN-2016-36   | 9          | 2016-08-23    | 2016-09-01  | 2016-09-11     |
| CC-MAIN-2016-30   | 8          | 2016-07-23    | 2016-07-31  | 2016-07-31     |
| CC-MAIN-2016-26   | 8          | 2016-06-24    | 2016-07-02  | 2016-07-03     |
| CC-MAIN-2016-22   | 8          | 2016-05-24    | 2016-06-01  | 2016-06-05     |
| CC-MAIN-2016-18   | 9          | 2016-04-28    | 2016-05-07  | 2016-05-08     |
| CC-MAIN-2016-07   | 10         | 2016-02-05    | 2016-02-15  | 2016-02-21     |
| CC-MAIN-2015-48   | 8          | 2015-11-24    | 2015-12-02  | 2015-12-06     |
| CC-MAIN-2015-40   | 10         | 2015-10-04    | 2015-10-14  | 2015-10-11     |
| CC-MAIN-2015-35   | 9          | 2015-08-27    | 2015-09-05  | 2015-09-06     |
| CC-MAIN-2015-32   | 8          | 2015-07-28    | 2015-08-05  | 2015-08-16     |
| CC-MAIN-2015-27   | 9          | 2015-06-29    | 2015-07-08  | 2015-07-12     |
| CC-MAIN-2015-22   | 13         | 2015-05-22    | 2015-06-04  | 2015-06-07     |
| CC-MAIN-2015-18   | 19         | 2015-04-18    | 2015-05-07  | 2015-05-10     |
| CC-MAIN-2015-14   | 7          | 2015-03-26    | 2015-04-02  | 2015-04-12     |
| CC-MAIN-2015-11   | 9          | 2015-02-26    | 2015-03-07  | 2015-03-22     |
| CC-MAIN-2015-06   | 8          | 2015-01-25    | 2015-02-02  | 2015-02-15     |
| CC-MAIN-2014-52   | 12         | 2014-12-17    | 2014-12-29  | 2015-01-04     |
| CC-MAIN-2014-49   | 9          | 2014-11-20    | 2014-11-29  | 2014-12-14     |
| CC-MAIN-2014-42   | 12         | 2014-10-20    | 2014-11-01  | 2014-10-26     |
| CC-MAIN-2014-41   | 17         | 2014-09-15    | 2014-10-02  | 2014-10-19     |
| CC-MAIN-2014-35   | 14         | 2014-08-20    | 2014-09-03  | 2014-09-07     |
| CC-MAIN-2014-23   | 24         | 2014-07-09    | 2014-08-02  | 2014-06-15     |
| CC-MAIN-2014-15   | 9          | 2014-04-16    | 2014-04-25  | 2014-04-20     |
| CC-MAIN-2014-10   | 10         | 2014-03-07    | 2014-03-17  | 2014-03-16     |
| CC-MAIN-2013-48   | 18         | 2013-12-04    | 2013-12-22  | 2013-12-08     |
| CC-MAIN-2013-20   | 33         | 2013-05-18    | 2013-06-20  | 2013-05-26     |
| CC-MAIN-2012      | 130        | 2012-01-27    | 2012-06-05  |                |
| CC-MAIN-2009-2010 | 435        | 2009-07-02    | 2010-09-10  |                |
| CC-MAIN-2008-2009 | 245        | 2008-05-09    | 2009-01-09  |                |



# Isn't it written somewhere?

This same problem is solved in CDX Toolkit where they [hardcode timestamps for early dates](https://github.com/cocrawler/cdx_toolkit/blob/0.9.33/cdx_toolkit/timeutils.py#L79):

```python
    table = {  # times provided by Sebastian
        '2012': timestamp_to_time('201206'),  # end 20120605, start was 20120127
        '2009-2010': timestamp_to_time('201009'),  # end 20100910, start was 20100910
        '2008-2009': timestamp_to_time('200901'),  # end 20090109, start was 20080509
    }
```

For more recent data which is in the form CRAWL-NAME-YYYY-WW they assume the first day of the week is at the end of the crawl.
Looking at recent crawls this is true; for example [the October 2021 Crawl](https://commoncrawl.org/2021/11/october-2021-crawl-archive-now-available/) was crawled Oct 15-28, and has label `CC-MAIN-2021-43`; which starts on 2021-10-31.
I've validated this by looking through the Common Crawl blog posts back to December 2020.

However it's not always true; looking at the [earliest crawl in this format](https://commoncrawl.org/2013/11/new-crawl-data-available/) which describes the format as:

> CRAWL-NAME-YYYY-WW – The name of the crawl and year + week# initiated on
>
> The 2013 wide web crawl data is located at /crawl-data/CC-MAIN-2013-20/ which represents the main CC crawl initiated during the 20th week of 2013.

Unfortunately I couldn't easily find blog releases with capture dates for every dataset, so I had to run the query to find out when this changed.
In fact it looks like it's all over the place before around mid 2014, and the ISO week start is pretty close but not always after the crawl.

| Index Name      | Earliest Date | Latest Date | ISO Week Start | Days After Week Start |
|:----------------|:--------------|:------------|:---------------|----------------------:|
| CC-MAIN-2017-17 | 2017-04-23    | 2017-05-01  | 2017-04-30     |                    -1 |
| CC-MAIN-2015-40 | 2015-10-04    | 2015-10-14  | 2015-10-11     |                    -3 |
| CC-MAIN-2014-42 | 2014-10-20    | 2014-11-01  | 2014-10-26     |                    -6 |
| CC-MAIN-2014-23 | 2014-07-09    | 2014-08-02  | 2014-06-15     |                   -48 |
| CC-MAIN-2014-15 | 2014-04-16    | 2014-04-25  | 2014-04-20     |                    -5 |
| CC-MAIN-2014-10 | 2014-03-07    | 2014-03-17  | 2014-03-16     |                    -1 |
| CC-MAIN-2013-48 | 2013-12-04    | 2013-12-22  | 2013-12-08     |                   -14 |
| CC-MAIN-2013-20 | 2013-05-18    | 2013-06-20  | 2013-05-26     |                   -25 |

# Missing index metadata

When running the queries I filtered to those with a 200 OK status, but got an error for exactly two indexes:

```
ERROR:root:404 Client Error: Not Found for url: https://index.commoncrawl.org/CC-MAIN-2015-11-index?url=en.wikipedia.org%2F%2A&output=json&filter=%3Dstatus%3A200
ERROR:root:404 Client Error: Not Found for url: https://index.commoncrawl.org/CC-MAIN-2015-06-index?url=en.wikipedia.org%2F%2A&output=json&filter=%3Dstatus%3A200
```

Running it without the filter showed *just* those two indexes seem to be missing status, and they are missing mimetype as well as other features.
For example here's the first two lines of a response from 2015-06:

```
[{'urlkey': 'org,wikipedia,en)/?banner=blackout',
  'timestamp': '20150131071342',
  'url': 'http://en.wikipedia.org/?banner=blackout',
  'digest': 'YZPQWVBHLGVUKY2DAZAVNZTMPOVOGV5P',
  'length': '17573',
  'offset': '107669292',
  'filename': 'crawl-data/CC-MAIN-2015-06/segments/1422122108378.68/warc/CC-MAIN-20150124175508-00190-ip-10-180-212-252.ec2.internal.warc.gz'},
 {'urlkey': 'org,wikipedia,en)/robots.txt',
  'timestamp': '20150129073755',
  'url': 'http://en.wikipedia.org/robots.txt',
  'digest': 'IXB3SAHXDS54OVWAQIIND22TH7HNXR7W',
  'length': '6017',
  'offset': '114218356',
  'filename': 'crawl-data/CC-MAIN-2015-06/segments/1422115855845.27/warc/CC-MAIN-20150124161055-00146-ip-10-180-212-252.ec2.internal.warc.gz'},
```

Compared with the previous index 2014-52:

```
{'urlkey': 'org,wikipedia,en)/?banner=blackout',
  'timestamp': '20141226070722',
  'url': 'http://en.wikipedia.org/?banner=blackout',
  'mime': 'text/html',
  'status': '200',
  'digest': 'CQ66DSSKOUHSJXUWYW3HI45BEG22ZTYI',
  'length': '18811',
  'offset': '33342669',
  'filename': 'crawl-data/CC-MAIN-2014-52/segments/1419447548645.134/warc/CC-MAIN-20141224185908-00083-ip-10-231-17-201.ec2.internal.warc.gz'},
 {'urlkey': 'org,wikipedia,en)/?title=pet_scanner',
  'timestamp': '20141227154942',
  'url': 'http://en.wikipedia.org/?title=PET_scanner',
  'mime': 'text/html',
  'status': '200',
  'digest': '52CNIQXTQ6DBUCSL7JQ7AUWP6CN3BU3L',
  'length': '49966',
  'offset': '30581136',
  'filename': 'crawl-data/CC-MAIN-2014-52/segments/1419447552326.50/warc/CC-MAIN-20141224185912-00056-ip-10-231-17-201.ec2.internal.warc.gz'},
```