---
categories:
- python
date: '2020-11-23T18:04:36+11:00'
image: /images/urllib3_retry.png
title: Retrying Python Requests
---

The computer networks that make up the internet are complex and handling an immense amount of traffic. 
So sometimes when you make a request it will fail intermittently, and you want to try until it succeeds.
This is easy in `requests` using `urllib3` `Retry`.

I was trying to download data from Common Crawl's S3 exports, but occasionally the process would fail due to a network or server error.
My process would keep the successful downloads using [an AtomicFileWriter](/atomic-writer), but I'd have to restart the process.
Here's what the new code looks like:


```
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
CC_DATA_URL = 'https://data.commoncrawl.org/'
RETRY_STRATEGY = Retry(
    total=5,
    backoff_factor=1
)
ADAPTER = HTTPAdapter(max_retries=RETRY_STRATEGY)
CC_HTTP = requests.Session()
CC_HTTP.mount(CC_DATA_URL, ADAPTER)

...
# was: response = requests.get(data_url, headers=headers)
response = CC_HTTP.get(data_url, headers=headers)
...
```

We use a [`requests.Session`](https://requests.readthedocs.io/en/latest/user/advanced/#session-objects) to be able to store the retry logic.
An advantage of Session is it reuses the TCP connections, and in my case made downloading twice as fast.

We then give the session information about how to retry on a domain.
We then store the information in a [HTTPAdapter](https://requests.readthedocs.io/en/latest/user/advanced/#transport-adapters) where we give it instructions on how to handle requests from a domain or protocol.
There we specify the [Retry strategy](https://urllib3.readthedocs.io/en/latest/reference/urllib3.util.html#module-urllib3.util.retry), in this case trying at most 5 times and exponentially backing off starting at 1 second. 

For a deeper dive into how to handle difficult cases with requests check out the [findwork.dev blog on advanced requests usage](https://findwork.dev/blog/advanced-usage-python-requests-timeouts-retries-hooks/), which also covers using timeouts (which is generally important but I haven't had an issue with yet).