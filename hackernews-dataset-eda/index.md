---
categories:
- python
- hnbooks
date: '2022-06-21T21:00:00+10:00'
image: /images/hackernews_eda.png
title: Hacker News Dataset EDA
---

> A mystery! A riddle! A puzzle! A quest! This was the moment that Ada loved best.
>
> *Ada Twist, Scientist*, Andrea Beaty

This is an exploration of 2021 [Hacker News](https://news.ycombinator.com/) posts as a precursor to [building a books dataset](https://skeptric.com/book-title-ner-outline/).

The data was sourced from the Google Bigquery public dataset `bigquery-public-data.hacker_news.full` using a [Kaggle notebook](https://www.kaggle.com/code/edwardjross/hackernews-2021-export/notebook).

```
SELECT *
FROM `bigquery-public-data.hacker_news.full`
where '2021-01-01' <= timestamp and timestamp < '2022-01-01'
```

I want to get a basic understanding of what's in the dataset before doing any data mining.

The [Hacker News FAQ](https://news.ycombinator.com/newsfaq.html) is useful for contextualising some of the fields.

This post was generated with a [Jupyter notebook](https://nbviewer.org/github/EdwardJRoss/skeptric/blob/master/static/notebooks/hackernews-dataset-eda.ipynb).

Please note that these comments may contain some explicit content.

# Load in data


```python
import numpy as np
import pandas as pd

import html

from pathlib import Path
```


```python
pd.options.display.max_columns = 100
```

Download [the data](https://www.kaggle.com/code/edwardjross/hackernews-2021-export/data) into this path first.

Make sure we use nullable dtypes to avoid converting integer identifier to floats, and set the unique `id` as the key.


```python
hn_path = Path('../data/hackernews2021.parquet')

df = pd.read_parquet(hn_path, use_nullable_dtypes=True).set_index('id')
```


```python
assert df.index.is_unique
```


```python
assert df.index.notna().all()
```

# Summary

Here's the schema described in Big Query

| name        | type      | description                                                           |
|-------------|-----------|-----------------------------------------------------------------------|
| title       | STRING    | Story title                                                           |
| url         | STRING    | Story url                                                             |
| text        | STRING    | Story or comment text                                                 |
| dead        | BOOLEAN   | Is dead?                                                              |
| by          | STRING    | The username of the item's author.                                    |
| score       | INTEGER   | Story score                                                           |
| time        | INTEGER   | Unix time                                                             |
| timestamp   | TIMESTAMP | Timestamp for the unix time                                           |
| type        | STRING    | Type of details (comment, comment_ranking, poll, story, job, pollopt) |
| id          | INTEGER   | The item's unique id.                                                 |
| parent      | INTEGER   | Parent comment ID                                                     |
| descendants | INTEGER   | Number of story or poll descendants                                   |
| ranking     | INTEGER   | Comment ranking                                                       |
| deleted     | BOOLEAN   | Is deleted?                                                           |



```python
df.dtypes
```




    title                       string
    url                         string
    text                        string
    dead                       boolean
    by                          string
    score                        Int64
    time                         Int64
    timestamp      datetime64[ns, UTC]
    type                        string
    parent                       Int64
    descendants                  Int64
    ranking                      Int64
    deleted                    boolean
    dtype: object



Here's a sample of the dataframe.

Note that we can view any individual item by appending the `id` in the URL `https://news.ycombinator.com/item?id=`


```python
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>url</th>
      <th>text</th>
      <th>dead</th>
      <th>by</th>
      <th>score</th>
      <th>time</th>
      <th>timestamp</th>
      <th>type</th>
      <th>parent</th>
      <th>descendants</th>
      <th>ranking</th>
      <th>deleted</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27405131</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>They didn&amp;#x27;t say they &lt;i&gt;weren&amp;#x27;t&lt;/i&gt; ...</td>
      <td>&lt;NA&gt;</td>
      <td>chrisseaton</td>
      <td>&lt;NA&gt;</td>
      <td>1622901869</td>
      <td>2021-06-05 14:04:29+00:00</td>
      <td>comment</td>
      <td>27405089</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>27814313</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>Check out &lt;a href="https:&amp;#x2F;&amp;#x2F;www.remno...</td>
      <td>&lt;NA&gt;</td>
      <td>noyesno</td>
      <td>&lt;NA&gt;</td>
      <td>1626119705</td>
      <td>2021-07-12 19:55:05+00:00</td>
      <td>comment</td>
      <td>27812726</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>28626089</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>Like a million-dollars pixel but with letters....</td>
      <td>&lt;NA&gt;</td>
      <td>alainchabat</td>
      <td>&lt;NA&gt;</td>
      <td>1632381114</td>
      <td>2021-09-23 07:11:54+00:00</td>
      <td>comment</td>
      <td>28626017</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>27143346</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>Not the question...</td>
      <td>&lt;NA&gt;</td>
      <td>SigmundA</td>
      <td>&lt;NA&gt;</td>
      <td>1620920426</td>
      <td>2021-05-13 15:40:26+00:00</td>
      <td>comment</td>
      <td>27143231</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>29053108</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>There’s the Unorganized Militia of the United ...</td>
      <td>&lt;NA&gt;</td>
      <td>User23</td>
      <td>&lt;NA&gt;</td>
      <td>1635636573</td>
      <td>2021-10-30 23:29:33+00:00</td>
      <td>comment</td>
      <td>29052087</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>27367848</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>Housing supply isn’t something that can’t chan...</td>
      <td>&lt;NA&gt;</td>
      <td>JCM9</td>
      <td>&lt;NA&gt;</td>
      <td>1622636746</td>
      <td>2021-06-02 12:25:46+00:00</td>
      <td>comment</td>
      <td>27367172</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>28052800</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>Final Fantasy XIV has been experiencing consta...</td>
      <td>&lt;NA&gt;</td>
      <td>amyjess</td>
      <td>&lt;NA&gt;</td>
      <td>1628017217</td>
      <td>2021-08-03 19:00:17+00:00</td>
      <td>comment</td>
      <td>28050798</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>28052805</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>How did you resolve it?</td>
      <td>&lt;NA&gt;</td>
      <td>8ytecoder</td>
      <td>&lt;NA&gt;</td>
      <td>1628017238</td>
      <td>2021-08-03 19:00:38+00:00</td>
      <td>comment</td>
      <td>28049375</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>26704924</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>This hasn&amp;#x27;t been my experience being vega...</td>
      <td>&lt;NA&gt;</td>
      <td>pacomerh</td>
      <td>&lt;NA&gt;</td>
      <td>1617657938</td>
      <td>2021-04-05 21:25:38+00:00</td>
      <td>comment</td>
      <td>26704794</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>27076885</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>Death services tread a very fine moral line.  ...</td>
      <td>&lt;NA&gt;</td>
      <td>curryst</td>
      <td>&lt;NA&gt;</td>
      <td>1620400897</td>
      <td>2021-05-07 15:21:37+00:00</td>
      <td>comment</td>
      <td>27075961</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
  </tbody>
</table>
<p>4155063 rows × 13 columns</p>
</div>



Every post has a time, timestamp and parent.

No post has a ranking.


```python
df.notna().mean().apply('{:0.2%}'.format)
```




    title            8.97%
    url              8.46%
    text            88.57%
    dead             3.87%
    by              97.22%
    score            9.04%
    time           100.00%
    timestamp      100.00%
    type           100.00%
    parent          90.64%
    descendants      7.00%
    ranking          0.00%
    deleted          2.78%
    dtype: object



We filtered to data in 2021, so it's all in this range


```python
df['timestamp'].min(), df['timestamp'].max()
```




    (Timestamp('2021-01-01 00:00:01+0000', tz='UTC'),
     Timestamp('2021-12-31 23:59:50+0000', tz='UTC'))



Most threads consist of a `story` which have `comments`. Apparently there are also `job` and `poll` objects.


```python
df['type'].value_counts()
```




    comment    3766009
    story       387194
    job           1422
    pollopt        385
    poll            53
    Name: type, dtype: Int64



# Date and Time

There's a spike in January (holidays?) a drop in February (lower days), but a fairly consistent amount of traffic.


```python
df['timestamp'].dt.month.value_counts().sort_index().plot()
```




    <AxesSubplot:>





![png](/post/hackernews-dataset-eda/output_20_1.png)



Looking at the daily traffic it look like there may be weekly effects, but aside from a spike towards the end of January it's fairly consistent.


```python
df['timestamp'].dt.date.value_counts().sort_index().plot()
```




    <AxesSubplot:>





![png](/post/hackernews-dataset-eda/output_22_1.png)



Most posts are made on the weekdays


```python
df['timestamp'].dt.day_name().value_counts()
```




    Tuesday      662106
    Wednesday    658830
    Thursday     654405
    Monday       628152
    Friday       625707
    Sunday       467553
    Saturday     458310
    Name: timestamp, dtype: int64



Based on the [4am rule](https://skeptric.com/4am-rule/) is looks like the most common timezone is around UTC-1.

This is slightly surprising, I would expect it could be closer to a US timezone (around -4 to -8). Maybe there's more posting from other regions than I'd have thought.


```python
df['timestamp'].dt.hour.value_counts().sort_index().plot()
```




    <AxesSubplot:>





![png](/post/hackernews-dataset-eda/output_26_1.png)



# Story

A story consists of a `title`, and it looks like either a `url` or `text`


```python
story = df.query('type=="story"')
story
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>url</th>
      <th>text</th>
      <th>dead</th>
      <th>by</th>
      <th>score</th>
      <th>time</th>
      <th>timestamp</th>
      <th>type</th>
      <th>parent</th>
      <th>descendants</th>
      <th>ranking</th>
      <th>deleted</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28540306</th>
      <td>CoinCircle for Life</td>
      <td>&lt;NA&gt;</td>
      <td>Hello, Lets join us to CoinCircle for our bett...</td>
      <td>True</td>
      <td>rend-airdrop</td>
      <td>1</td>
      <td>1631719412</td>
      <td>2021-09-15 15:23:32+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>26273978</th>
      <td>Find the number of third-party privacy tracker...</td>
      <td>&lt;NA&gt;</td>
      <td>Exodus Privacy is a non-profit organization th...</td>
      <td>True</td>
      <td>moulidorai</td>
      <td>1</td>
      <td>1614341393</td>
      <td>2021-02-26 12:09:53+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>27214431</th>
      <td>Ask HN: Desk Recommendations?</td>
      <td>&lt;NA&gt;</td>
      <td>I often see standing desk recommendations here...</td>
      <td>True</td>
      <td>throwaw9l938ni</td>
      <td>1</td>
      <td>1621458219</td>
      <td>2021-05-19 21:03:39+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>25705820</th>
      <td>Demand Hunter Biden Be Arrested</td>
      <td>&lt;NA&gt;</td>
      <td>There are so many pictures of Hunter Biden, Jo...</td>
      <td>True</td>
      <td>bidenpedo</td>
      <td>1</td>
      <td>1610232470</td>
      <td>2021-01-09 22:47:50+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>26110009</th>
      <td>Deep learning multivariate nonlinear regression</td>
      <td>&lt;NA&gt;</td>
      <td>Does deep learning really work for regression ...</td>
      <td>True</td>
      <td>dl_regression</td>
      <td>1</td>
      <td>1613095333</td>
      <td>2021-02-12 02:02:13+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>28773509</th>
      <td>Apple to face EU antitrust charge over NFC chip</td>
      <td>https://www.reuters.com/technology/exclusive-e...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>nojito</td>
      <td>170</td>
      <td>1633530062</td>
      <td>2021-10-06 14:21:02+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>219</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>26400239</th>
      <td>The Roblox Microverse</td>
      <td>https://stratechery.com/2021/the-roblox-microv...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>Kinrany</td>
      <td>173</td>
      <td>1615306495</td>
      <td>2021-03-09 16:14:55+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>203</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>27559832</th>
      <td>Safari 15 on Mac OS, a user interface mess</td>
      <td>https://morrick.me/archives/9368</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>freediver</td>
      <td>463</td>
      <td>1624104913</td>
      <td>2021-06-19 12:15:13+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>353</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>26992205</th>
      <td>Stock Market Returns Are Anything but Average</td>
      <td>https://awealthofcommonsense.com/2021/04/stock...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>RickJWagner</td>
      <td>222</td>
      <td>1619783307</td>
      <td>2021-04-30 11:48:27+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>413</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>29738298</th>
      <td>Tokyo police lose 2 floppy disks containing in...</td>
      <td>https://mainichi.jp/english/articles/20211227/...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>ardel95</td>
      <td>232</td>
      <td>1640883038</td>
      <td>2021-12-30 16:50:38+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>218</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
  </tbody>
</table>
<p>387194 rows × 13 columns</p>
</div>



Stories normally have title and a URL, and occasionally have text.

They're almost always `by` someone, and have a `score`.
They never have a `parent` (they're always top level), but they normally have `descendants`.

Some are dead (removed by Hacker News) and some are deleted (removed by the author).


```python
(
    story
    .notna()
    .mean()
    .apply('{:0.1%}'.format)
)
```




    title           95.9%
    url             90.5%
    text             4.9%
    dead            22.5%
    by              96.6%
    score           96.6%
    time           100.0%
    timestamp      100.0%
    type           100.0%
    parent           0.0%
    descendants     75.1%
    ranking          0.0%
    deleted          3.4%
    dtype: object



By seems to be missing only for deleted stories


```python
(
    story
    .query('by.isna()')
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>url</th>
      <th>text</th>
      <th>dead</th>
      <th>by</th>
      <th>score</th>
      <th>time</th>
      <th>timestamp</th>
      <th>type</th>
      <th>parent</th>
      <th>descendants</th>
      <th>ranking</th>
      <th>deleted</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>26779931</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>1618238390</td>
      <td>2021-04-12 14:39:50+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>True</td>
    </tr>
    <tr>
      <th>26122158</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>1613203434</td>
      <td>2021-02-13 08:03:54+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>True</td>
    </tr>
    <tr>
      <th>25699401</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>1610190538</td>
      <td>2021-01-09 11:08:58+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>True</td>
    </tr>
    <tr>
      <th>26206857</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>1613848074</td>
      <td>2021-02-20 19:07:54+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>True</td>
    </tr>
    <tr>
      <th>26316571</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>1614700390</td>
      <td>2021-03-02 15:53:10+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>True</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>28201589</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>1629140598</td>
      <td>2021-08-16 19:03:18+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>True</td>
    </tr>
    <tr>
      <th>26786548</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>1618271177</td>
      <td>2021-04-12 23:46:17+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>True</td>
    </tr>
    <tr>
      <th>26689984</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>1617548611</td>
      <td>2021-04-04 15:03:31+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>True</td>
    </tr>
    <tr>
      <th>27349809</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>1622509992</td>
      <td>2021-06-01 01:13:12+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>True</td>
    </tr>
    <tr>
      <th>25913791</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>1611651379</td>
      <td>2021-01-26 08:56:19+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>True</td>
    </tr>
  </tbody>
</table>
<p>13300 rows × 13 columns</p>
</div>



Every story has a `by` unless it's deleted or dead.


```python
(
    story
    .query('by.isna() & deleted.isna() & dead.isna()')
)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>url</th>
      <th>text</th>
      <th>dead</th>
      <th>by</th>
      <th>score</th>
      <th>time</th>
      <th>timestamp</th>
      <th>type</th>
      <th>parent</th>
      <th>descendants</th>
      <th>ranking</th>
      <th>deleted</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



>  **How do I make a link in a text submission?**
>
> You can't. This is to prevent people from submitting a link with their comments in a privileged position at the top of the page. If you want to submit a link with comments, just submit it, then add a regular comment.

This seems to be true most of the time


```python
(
    story
    .assign(
        has_url = lambda _: ~_.url.isna(),
        has_text = lambda _: ~_.text.isna(),
        has_url_and_text = lambda _: _.has_url & _.has_text,
        has_url_or_text = lambda _: _.has_url | _.has_text,
    )
    .filter(like='has_')
    .mean()
)
```




    has_url             0.904606
    has_text            0.048536
    has_url_and_text    0.000031
    has_url_or_text     0.953111
    dtype: float64



There seems to be a few exceptions for Show HN.

We actually don't have metadata to identify Ask HN and Show HN.


```python
story.query('~url.isna() & ~text.isna()')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>url</th>
      <th>text</th>
      <th>dead</th>
      <th>by</th>
      <th>score</th>
      <th>time</th>
      <th>timestamp</th>
      <th>type</th>
      <th>parent</th>
      <th>descendants</th>
      <th>ranking</th>
      <th>deleted</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28074827</th>
      <td>Show HN: Visualizing a Codebase</td>
      <td>https://octo.github.com/projects/repo-visualiz...</td>
      <td>I explored an alternative way to view codebase...</td>
      <td>&lt;NA&gt;</td>
      <td>wattenberger</td>
      <td>283</td>
      <td>1628176192</td>
      <td>2021-08-05 15:09:52+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>96</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>29019925</th>
      <td>Show HN: Guestio – A better way to find and bo...</td>
      <td>https://guestio.com/</td>
      <td>Guestio is an all-in-one tool designed to help...</td>
      <td>&lt;NA&gt;</td>
      <td>travischappelll</td>
      <td>4</td>
      <td>1635374411</td>
      <td>2021-10-27 22:40:11+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>2</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>26346586</th>
      <td>Show HN: Practical Python Projects book release</td>
      <td>https://practicalpython.yasoob.me</td>
      <td>Hi everyone!&lt;p&gt;I just released the Practical P...</td>
      <td>&lt;NA&gt;</td>
      <td>yasoob</td>
      <td>88</td>
      <td>1614884336</td>
      <td>2021-03-04 18:58:56+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>14</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>27787426</th>
      <td>Show HN: Homer – A tool to build interactive t...</td>
      <td>https://usehomer.app</td>
      <td>Hi HN, my name is Rahul Sarathy and I built Ho...</td>
      <td>&lt;NA&gt;</td>
      <td>Outofthebot</td>
      <td>62</td>
      <td>1625858111</td>
      <td>2021-07-09 19:15:11+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>26</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>27684916</th>
      <td>Why do we work so damn much?</td>
      <td>https://www.nytimes.com/2021/06/29/opinion/ezr...</td>
      <td>The New York Times: Opinion | Why Do We Work S...</td>
      <td>&lt;NA&gt;</td>
      <td>anirudhgarg</td>
      <td>44</td>
      <td>1625027907</td>
      <td>2021-06-30 04:38:27+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>62</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>27257586</th>
      <td>C is not a serious programming language</td>
      <td>https://www.yodaiken.com/2021/05/16/c-is-not-a...</td>
      <td>&amp;lt;https:&amp;#x2F;&amp;#x2F;www.yodaiken.com&amp;#x2F;20...</td>
      <td>True</td>
      <td>vyodaiken</td>
      <td>1</td>
      <td>1621796527</td>
      <td>2021-05-23 19:02:07+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>28934833</th>
      <td>Bioelektryczność – Polish Robotics (1968) [video]</td>
      <td>https://www.youtube.com/watch?v=NjrYk546uBA</td>
      <td>I&amp;#x27;m curious what was the state of an art ...</td>
      <td>&lt;NA&gt;</td>
      <td>danielEM</td>
      <td>134</td>
      <td>1634757119</td>
      <td>2021-10-20 19:11:59+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>28</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>26998308</th>
      <td>Show HN: Second-Chance Pool</td>
      <td>https://news.ycombinator.com/pool</td>
      <td>HN&amp;#x27;s second-chance pool is a way to give ...</td>
      <td>&lt;NA&gt;</td>
      <td>dang</td>
      <td>543</td>
      <td>1619811719</td>
      <td>2021-04-30 19:41:59+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>91</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>29225588</th>
      <td>Show HN: Grapic – Real whiteboards online usin...</td>
      <td>https://www.grapic.co/</td>
      <td>Hi HN,&lt;p&gt;During the pandemic, two friends and ...</td>
      <td>&lt;NA&gt;</td>
      <td>nikonp</td>
      <td>97</td>
      <td>1636969643</td>
      <td>2021-11-15 09:47:23+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>24</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>29705761</th>
      <td>Diego Rivera’s Vaccine Mural in Detroit in the...</td>
      <td>https://historyofvaccines.blog/2021/07/12/dieg...</td>
      <td>https:&amp;#x2F;&amp;#x2F;historyofvaccines.blog&amp;#x2F;...</td>
      <td>&lt;NA&gt;</td>
      <td>barbe</td>
      <td>4</td>
      <td>1640632550</td>
      <td>2021-12-27 19:15:50+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>1</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>26251143</th>
      <td>My experience as a Gazan girl getting into Sil...</td>
      <td>https://daliaawad28.medium.com/my-experience-a...</td>
      <td>Hiii everyone, this is my first time posting h...</td>
      <td>&lt;NA&gt;</td>
      <td>daliaawad</td>
      <td>1723</td>
      <td>1614181663</td>
      <td>2021-02-24 15:47:43+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>460</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>29655974</th>
      <td>Show HN: Jig – a tool to define, compute and m...</td>
      <td>https://www.jigdev.com</td>
      <td>Hi HN,&lt;p&gt;8 months ago, I posted “Ask HN: I bui...</td>
      <td>&lt;NA&gt;</td>
      <td>d--b</td>
      <td>74</td>
      <td>1640210325</td>
      <td>2021-12-22 21:58:45+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>24</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
  </tbody>
</table>
</div>



The scores look like they follow a sort of power law.


```python
(
    story
    .query('dead.isna() & deleted.isna()')
    .score
    .fillna(0.)
    .plot
    .hist(logy=True, bins=40)
)
```




    <AxesSubplot:ylabel='Frequency'>





![png](/post/hackernews-dataset-eda/output_42_1.png)



And descendants follow a similar path


```python
(
    story
    .query('dead.isna() & deleted.isna()')
    .descendants
    .fillna(0.)
    .plot
    .hist(logy=True, bins=40)
)
```




    <AxesSubplot:ylabel='Frequency'>





![png](/post/hackernews-dataset-eda/output_44_1.png)



It looks like the titles must be below around 80 characters and are typically around 60


```python
story.title.fillna('').str.len().plot.hist(bins=20)
```




    <AxesSubplot:ylabel='Frequency'>





![png](/post/hackernews-dataset-eda/output_46_1.png)



The text can be much longer and follows a decaying distribution


```python
story.text.fillna('').str.len().plot.hist(bins=20, logy=True)
```




    <AxesSubplot:ylabel='Frequency'>





![png](/post/hackernews-dataset-eda/output_48_1.png)



Some URLs can be *very* long (I guess they can have all sorts of query parameters)


```python
story.url.fillna('').str.len().plot.hist(bins=20, logy=True)
```




    <AxesSubplot:ylabel='Frequency'>





![png](/post/hackernews-dataset-eda/output_50_1.png)




```python
from urllib.parse import urlparse
```

Common hosts; Github, YouTube, twitter


```python
story_url_host_counts = story['url'].dropna().map(lambda x: urlparse(x).hostname).value_counts()

story_url_host_counts.head(20)
```




    github.com                13622
    www.youtube.com           12843
    twitter.com                6968
    en.wikipedia.org           6218
    www.nytimes.com            5647
    medium.com                 4964
    www.theguardian.com        4244
    arstechnica.com            3545
    www.bloomberg.com          3007
    www.bbc.com                2996
    www.theverge.com           2888
    dev.to                     2746
    www.wsj.com                2704
    www.reuters.com            2445
    techcrunch.com             1820
    www.cnbc.com               1792
    www.reddit.com             1430
    www.bbc.co.uk              1426
    www.washingtonpost.com     1413
    www.theatlantic.com        1374
    Name: url, dtype: int64



Again a small handful of hosts get most of the links


```python
story_url_host_counts.plot.hist(logy=True, bins=20)
```




    <AxesSubplot:ylabel='Frequency'>





![png](/post/hackernews-dataset-eda/output_55_1.png)



There are some power users that post a *lot* of stories


```python
story_by_counts = story.by.value_counts()

story_by_counts.head(20)
```




    Tomte              4856
    todsacerdoti       3031
    tosh               2940
    pseudolus          2876
    rbanffy            2875
    mooreds            1915
    samizdis           1834
    giuliomagnifico    1570
    feross             1491
    CapitalistCartr    1413
    ingve              1399
    fortran77          1358
    gmays              1162
    infodocket         1098
    belter             1078
    graderjs           1061
    elsewhen           1053
    kiyanwang          1009
    1cvmask            1005
    LinuxBender         996
    Name: by, dtype: Int64



And again a fast decline


```python
story_by_counts.plot.hist(logy=True, bins=20)
```




    <AxesSubplot:ylabel='Frequency'>





![png](/post/hackernews-dataset-eda/output_59_1.png)



# Comments


```python
comments = df.query('type == "comment"')
comments
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>url</th>
      <th>text</th>
      <th>dead</th>
      <th>by</th>
      <th>score</th>
      <th>time</th>
      <th>timestamp</th>
      <th>type</th>
      <th>parent</th>
      <th>descendants</th>
      <th>ranking</th>
      <th>deleted</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27405131</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>They didn&amp;#x27;t say they &lt;i&gt;weren&amp;#x27;t&lt;/i&gt; ...</td>
      <td>&lt;NA&gt;</td>
      <td>chrisseaton</td>
      <td>&lt;NA&gt;</td>
      <td>1622901869</td>
      <td>2021-06-05 14:04:29+00:00</td>
      <td>comment</td>
      <td>27405089</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>27814313</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>Check out &lt;a href="https:&amp;#x2F;&amp;#x2F;www.remno...</td>
      <td>&lt;NA&gt;</td>
      <td>noyesno</td>
      <td>&lt;NA&gt;</td>
      <td>1626119705</td>
      <td>2021-07-12 19:55:05+00:00</td>
      <td>comment</td>
      <td>27812726</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>28626089</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>Like a million-dollars pixel but with letters....</td>
      <td>&lt;NA&gt;</td>
      <td>alainchabat</td>
      <td>&lt;NA&gt;</td>
      <td>1632381114</td>
      <td>2021-09-23 07:11:54+00:00</td>
      <td>comment</td>
      <td>28626017</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>27143346</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>Not the question...</td>
      <td>&lt;NA&gt;</td>
      <td>SigmundA</td>
      <td>&lt;NA&gt;</td>
      <td>1620920426</td>
      <td>2021-05-13 15:40:26+00:00</td>
      <td>comment</td>
      <td>27143231</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>29053108</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>There’s the Unorganized Militia of the United ...</td>
      <td>&lt;NA&gt;</td>
      <td>User23</td>
      <td>&lt;NA&gt;</td>
      <td>1635636573</td>
      <td>2021-10-30 23:29:33+00:00</td>
      <td>comment</td>
      <td>29052087</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>27367848</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>Housing supply isn’t something that can’t chan...</td>
      <td>&lt;NA&gt;</td>
      <td>JCM9</td>
      <td>&lt;NA&gt;</td>
      <td>1622636746</td>
      <td>2021-06-02 12:25:46+00:00</td>
      <td>comment</td>
      <td>27367172</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>28052800</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>Final Fantasy XIV has been experiencing consta...</td>
      <td>&lt;NA&gt;</td>
      <td>amyjess</td>
      <td>&lt;NA&gt;</td>
      <td>1628017217</td>
      <td>2021-08-03 19:00:17+00:00</td>
      <td>comment</td>
      <td>28050798</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>28052805</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>How did you resolve it?</td>
      <td>&lt;NA&gt;</td>
      <td>8ytecoder</td>
      <td>&lt;NA&gt;</td>
      <td>1628017238</td>
      <td>2021-08-03 19:00:38+00:00</td>
      <td>comment</td>
      <td>28049375</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>26704924</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>This hasn&amp;#x27;t been my experience being vega...</td>
      <td>&lt;NA&gt;</td>
      <td>pacomerh</td>
      <td>&lt;NA&gt;</td>
      <td>1617657938</td>
      <td>2021-04-05 21:25:38+00:00</td>
      <td>comment</td>
      <td>26704794</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>27076885</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>Death services tread a very fine moral line.  ...</td>
      <td>&lt;NA&gt;</td>
      <td>curryst</td>
      <td>&lt;NA&gt;</td>
      <td>1620400897</td>
      <td>2021-05-07 15:21:37+00:00</td>
      <td>comment</td>
      <td>27075961</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
  </tbody>
</table>
<p>3766009 rows × 13 columns</p>
</div>



Comments can't have a tile or a URL.

They almost always have a `text` and a `by` (I'd guess it's missing for deleted and dead threads).

We don't ever get a `score` or `ranking` or `descendants` even though these things may make sense.


```python
(
    comments
    .notna()
    .mean()
    .apply('{:0.1%}'.format)
)
```




    title            0.0%
    url              0.0%
    text            97.2%
    dead             2.0%
    by              97.3%
    score            0.0%
    time           100.0%
    timestamp      100.0%
    type           100.0%
    parent         100.0%
    descendants      0.0%
    ranking          0.0%
    deleted          2.7%
    dtype: object



# Parents

We can look at the type of the parent's comments (they'll sometimes be missing if the parent was posted before our cutoff date.

Most comments parent is another comment in a thread.


```python
(
    comments
    .merge(df['type'], how='left', left_on='parent', right_index=True, suffixes=('', '_parent'), validate='m:1')
    ['type_parent']
    .value_counts(dropna=False)
)
```




    comment    2997792
    story       765342
    <NA>          2412
    poll           463
    Name: type_parent, dtype: Int64



We can efficiently look up a parent using a dictionary, returning `<NA>` when it's not there.


```python
from collections import defaultdict

parent_dict = df['parent'].dropna().to_dict()

parent_dict = defaultdict(lambda: pd.NA, parent_dict)

```


```python
%%time
df['parent'].map(parent_dict, na_action='ignore')
```

    CPU times: user 2.4 s, sys: 28.1 ms, total: 2.43 s
    Wall time: 2.43 s





    id
    27405131    27405024
    27814313    27807850
    28626089    28625485
    27143346    27142955
    29053108    29052012
                  ...
    27367848        <NA>
    28052800    28049873
    28052805    28046997
    26704924    26704392
    27076885    27074332
    Name: parent, Length: 4155063, dtype: object



We can do this iteratively to find all the parents.

When there is no parent we'll return `<NA>`; this particular way of doing it gets faster the fewer non-null elements there are.


```python
from tqdm.notebook import tqdm

MAX_DEPTH = 50

df['parent0'] = df['parent']

for idx in tqdm(range(MAX_DEPTH)):
    last_col = f'parent{idx}'
    col = f'parent{idx+1}'

    df[col] = df[last_col].map(parent_dict, na_action='ignore')
    if df[col].isna().all():
        del df[col]
        break

```


      0%|          | 0/50 [00:00<?, ?it/s]


We can now see all the parents of any element


```python
df.filter(regex='parent\d+')
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>parent0</th>
      <th>parent1</th>
      <th>parent2</th>
      <th>parent3</th>
      <th>parent4</th>
      <th>parent5</th>
      <th>parent6</th>
      <th>parent7</th>
      <th>parent8</th>
      <th>parent9</th>
      <th>parent10</th>
      <th>parent11</th>
      <th>parent12</th>
      <th>parent13</th>
      <th>parent14</th>
      <th>parent15</th>
      <th>parent16</th>
      <th>parent17</th>
      <th>parent18</th>
      <th>parent19</th>
      <th>parent20</th>
      <th>parent21</th>
      <th>parent22</th>
      <th>parent23</th>
      <th>parent24</th>
      <th>parent25</th>
      <th>parent26</th>
      <th>parent27</th>
      <th>parent28</th>
      <th>parent29</th>
      <th>parent30</th>
      <th>parent31</th>
      <th>parent32</th>
      <th>parent33</th>
      <th>parent34</th>
      <th>parent35</th>
      <th>parent36</th>
      <th>parent37</th>
      <th>parent38</th>
      <th>parent39</th>
      <th>parent40</th>
      <th>parent41</th>
      <th>parent42</th>
      <th>parent43</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27405131</th>
      <td>27405089</td>
      <td>27405024</td>
      <td>27404902</td>
      <td>27404548</td>
      <td>27404512</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>27814313</th>
      <td>27812726</td>
      <td>27807850</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>28626089</th>
      <td>28626017</td>
      <td>28625485</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>27143346</th>
      <td>27143231</td>
      <td>27142955</td>
      <td>27142884</td>
      <td>27142567</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>29053108</th>
      <td>29052087</td>
      <td>29052012</td>
      <td>29051947</td>
      <td>29051758</td>
      <td>29051607</td>
      <td>29051478</td>
      <td>29051448</td>
      <td>29051365</td>
      <td>29051109</td>
      <td>29043296</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>27367848</th>
      <td>27367172</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>28052800</th>
      <td>28050798</td>
      <td>28049873</td>
      <td>28049688</td>
      <td>28049620</td>
      <td>28049359</td>
      <td>28048919</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>28052805</th>
      <td>28049375</td>
      <td>28046997</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>26704924</th>
      <td>26704794</td>
      <td>26704392</td>
      <td>26703874</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
    <tr>
      <th>27076885</th>
      <td>27075961</td>
      <td>27074332</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
    </tr>
  </tbody>
</table>
<p>4155063 rows × 44 columns</p>
</div>



One useful concept is the *root*, the parent that has no parents itself (generally because it's top level, but sometimes because the parent isn't in the dataframe).


```python
%%time
root = None

for col in df.filter(regex='parent\d+').iloc[:,::-1]:
    if root is None:
        root = df[col]
    else:
        root = root.combine_first(df[col])
df['root'] = root
```

    CPU times: user 11.1 s, sys: 826 ms, total: 11.9 s
    Wall time: 11.9 s


We can also get the depth; how parents does it have?


```python
df['depth'] = df.filter(regex='parent\d+').notna().sum(axis=1)
```

What's the distribution of depth for comments?


```python
comments = df.query('type=="comment"')
```

That's some kind of zero-inflated distribution.


```python
comments['depth'].value_counts().plot(logy=True)
```




    <AxesSubplot:>





![png](/post/hackernews-dataset-eda/output_81_1.png)



We can check the type of the root (we get `<NA>` when it's not in the tree).

The vast majority of the the root of a comment is a story.


```python
df.merge(comments['root'], left_index=True, right_on='root', how='right')['type'].value_counts(dropna=False)
```




    story    3759475
    <NA>        5181
    poll        1353
    Name: type, dtype: Int64



Let's compare the `descendants` column with the


```python
stories = df.query('type=="story"')
```


```python
df['root'].value_counts()
```




    25706993    4029
    28693060    3088
    25661474    2638
    26347654    2372
    26487854    2155
                ...
    27038587       1
    26640257       1
    28404872       1
    27531105       1
    28347619       1
    Name: root, Length: 121760, dtype: int64



They're highly correlated with some outliers near zero.

Some reasons I can think they would differ:

* Time Filter - we may miss some comments made after the time cutoff (would make descendants > children)
* Time of capture - there may be some uncounted descendants if they were captured before children (would make descendants < children)
* Exclusions - descendants may not be counted if they are dead or deleted (would make descendants < children)


```python
children_counts = comments.loc[comments['dead'].isna() & comments['deleted'].isna(), 'root'].value_counts().rename('children')

children_counts = pd.concat([stories['descendants'], comments.loc[comments['dead'].isna() & comments['deleted'].isna(), 'root'].value_counts().rename('children')], axis=1).fillna(0)

children_counts.plot.scatter('descendants', 'children')
```




    <AxesSubplot:xlabel='descendants', ylabel='children'>





![png](/post/hackernews-dataset-eda/output_88_1.png)




```python
children_counts['diff'] = children_counts['descendants'] - children_counts['children']

children_counts.plot.scatter('descendants', 'diff')
```




    <AxesSubplot:xlabel='descendants', ylabel='diff'>





![png](/post/hackernews-dataset-eda/output_89_1.png)



The cases where descendants >> children they were posted near our cutoff date, the end of 2021.


```python
children_counts[children_counts['diff'] > 100]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>descendants</th>
      <th>children</th>
      <th>diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29752379</th>
      <td>155</td>
      <td>0.0</td>
      <td>155.0</td>
    </tr>
    <tr>
      <th>29749123</th>
      <td>126</td>
      <td>0.0</td>
      <td>126.0</td>
    </tr>
    <tr>
      <th>29753218</th>
      <td>207</td>
      <td>63.0</td>
      <td>144.0</td>
    </tr>
    <tr>
      <th>29753513</th>
      <td>130</td>
      <td>0.0</td>
      <td>130.0</td>
    </tr>
    <tr>
      <th>29753183</th>
      <td>275</td>
      <td>26.0</td>
      <td>249.0</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.loc[children_counts[children_counts['diff'] > 100].index]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>url</th>
      <th>text</th>
      <th>dead</th>
      <th>by</th>
      <th>score</th>
      <th>time</th>
      <th>timestamp</th>
      <th>type</th>
      <th>parent</th>
      <th>descendants</th>
      <th>ranking</th>
      <th>deleted</th>
      <th>parent0</th>
      <th>parent1</th>
      <th>parent2</th>
      <th>parent3</th>
      <th>parent4</th>
      <th>parent5</th>
      <th>parent6</th>
      <th>parent7</th>
      <th>parent8</th>
      <th>parent9</th>
      <th>parent10</th>
      <th>parent11</th>
      <th>parent12</th>
      <th>parent13</th>
      <th>parent14</th>
      <th>parent15</th>
      <th>parent16</th>
      <th>parent17</th>
      <th>parent18</th>
      <th>parent19</th>
      <th>parent20</th>
      <th>parent21</th>
      <th>parent22</th>
      <th>parent23</th>
      <th>parent24</th>
      <th>parent25</th>
      <th>parent26</th>
      <th>parent27</th>
      <th>parent28</th>
      <th>parent29</th>
      <th>parent30</th>
      <th>parent31</th>
      <th>parent32</th>
      <th>parent33</th>
      <th>parent34</th>
      <th>parent35</th>
      <th>parent36</th>
      <th>parent37</th>
      <th>parent38</th>
      <th>parent39</th>
      <th>parent40</th>
      <th>parent41</th>
      <th>parent42</th>
      <th>parent43</th>
      <th>root</th>
      <th>depth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>29752379</th>
      <td>A Guide to Twitter</td>
      <td>https://tasshin.com/blog/a-guide-to-twitter/</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>mwfogleman</td>
      <td>228</td>
      <td>1640983643</td>
      <td>2021-12-31 20:47:23+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>155</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29749123</th>
      <td>Safest mushrooms to forage and eat</td>
      <td>https://www.fieldandstream.com/story/survival/...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>mizzao</td>
      <td>167</td>
      <td>1640965909</td>
      <td>2021-12-31 15:51:49+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>126</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29753218</th>
      <td>Why Brahmins lead Western firms but rarely Ind...</td>
      <td>https://www.economist.com/asia/2022/01/01/why-...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>pseudolus</td>
      <td>141</td>
      <td>1640990143</td>
      <td>2021-12-31 22:35:43+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>207</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29753513</th>
      <td>If – A Poem by Rudyard Kipling</td>
      <td>https://poets.org/poem/if</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>BrindleBox</td>
      <td>282</td>
      <td>1640992493</td>
      <td>2021-12-31 23:14:53+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>130</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
    </tr>
    <tr>
      <th>29753183</th>
      <td>Belgian scientific base in Antarctica engulfed...</td>
      <td>https://www.brusselstimes.com/belgium-all-news...</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>justinzollars</td>
      <td>227</td>
      <td>1640989917</td>
      <td>2021-12-31 22:31:57+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>275</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



All the cases that error on this side there are no descendants


```python
children_counts[children_counts['diff'] < -100]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>descendants</th>
      <th>children</th>
      <th>diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28733467</th>
      <td>0</td>
      <td>197.0</td>
      <td>-197.0</td>
    </tr>
    <tr>
      <th>28761974</th>
      <td>0</td>
      <td>247.0</td>
      <td>-247.0</td>
    </tr>
    <tr>
      <th>28752512</th>
      <td>0</td>
      <td>297.0</td>
      <td>-297.0</td>
    </tr>
    <tr>
      <th>25669864</th>
      <td>0</td>
      <td>946.0</td>
      <td>-946.0</td>
    </tr>
    <tr>
      <th>25594068</th>
      <td>0</td>
      <td>329.0</td>
      <td>-329.0</td>
    </tr>
    <tr>
      <th>25598606</th>
      <td>0</td>
      <td>230.0</td>
      <td>-230.0</td>
    </tr>
    <tr>
      <th>25598768</th>
      <td>0</td>
      <td>191.0</td>
      <td>-191.0</td>
    </tr>
    <tr>
      <th>25597891</th>
      <td>0</td>
      <td>184.0</td>
      <td>-184.0</td>
    </tr>
    <tr>
      <th>25591202</th>
      <td>0</td>
      <td>153.0</td>
      <td>-153.0</td>
    </tr>
    <tr>
      <th>25590022</th>
      <td>0</td>
      <td>129.0</td>
      <td>-129.0</td>
    </tr>
    <tr>
      <th>25732809</th>
      <td>0</td>
      <td>127.0</td>
      <td>-127.0</td>
    </tr>
  </tbody>
</table>
</div>



There are a few cases where it's not in the index at all (maybe the story was posted just before the cutoff? we could confirm this with the children dates)


```python
children_counts[children_counts['diff'] < -100][~children_counts[children_counts['diff'] < -100].index.isin(df.index)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>descendants</th>
      <th>children</th>
      <th>diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25594068</th>
      <td>0</td>
      <td>329.0</td>
      <td>-329.0</td>
    </tr>
    <tr>
      <th>25598606</th>
      <td>0</td>
      <td>230.0</td>
      <td>-230.0</td>
    </tr>
    <tr>
      <th>25598768</th>
      <td>0</td>
      <td>191.0</td>
      <td>-191.0</td>
    </tr>
    <tr>
      <th>25597891</th>
      <td>0</td>
      <td>184.0</td>
      <td>-184.0</td>
    </tr>
    <tr>
      <th>25591202</th>
      <td>0</td>
      <td>153.0</td>
      <td>-153.0</td>
    </tr>
    <tr>
      <th>25590022</th>
      <td>0</td>
      <td>129.0</td>
      <td>-129.0</td>
    </tr>
  </tbody>
</table>
</div>



For the others


```python
children_counts[(children_counts['diff'] < -100) & children_counts.index.isin(df.index)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>descendants</th>
      <th>children</th>
      <th>diff</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28733467</th>
      <td>0</td>
      <td>197.0</td>
      <td>-197.0</td>
    </tr>
    <tr>
      <th>28761974</th>
      <td>0</td>
      <td>247.0</td>
      <td>-247.0</td>
    </tr>
    <tr>
      <th>28752512</th>
      <td>0</td>
      <td>297.0</td>
      <td>-297.0</td>
    </tr>
    <tr>
      <th>25669864</th>
      <td>0</td>
      <td>946.0</td>
      <td>-946.0</td>
    </tr>
    <tr>
      <th>25732809</th>
      <td>0</td>
      <td>127.0</td>
      <td>-127.0</td>
    </tr>
  </tbody>
</table>
</div>



Most of the time they differ its because the story is dead.


```python
df.loc[children_counts[(children_counts['diff'] < -100) & children_counts.index.isin(df.index)].index]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>url</th>
      <th>text</th>
      <th>dead</th>
      <th>by</th>
      <th>score</th>
      <th>time</th>
      <th>timestamp</th>
      <th>type</th>
      <th>parent</th>
      <th>descendants</th>
      <th>ranking</th>
      <th>deleted</th>
      <th>parent0</th>
      <th>parent1</th>
      <th>parent2</th>
      <th>parent3</th>
      <th>parent4</th>
      <th>parent5</th>
      <th>parent6</th>
      <th>parent7</th>
      <th>parent8</th>
      <th>parent9</th>
      <th>parent10</th>
      <th>parent11</th>
      <th>parent12</th>
      <th>parent13</th>
      <th>parent14</th>
      <th>parent15</th>
      <th>parent16</th>
      <th>parent17</th>
      <th>parent18</th>
      <th>parent19</th>
      <th>parent20</th>
      <th>parent21</th>
      <th>parent22</th>
      <th>parent23</th>
      <th>parent24</th>
      <th>parent25</th>
      <th>parent26</th>
      <th>parent27</th>
      <th>parent28</th>
      <th>parent29</th>
      <th>parent30</th>
      <th>parent31</th>
      <th>parent32</th>
      <th>parent33</th>
      <th>parent34</th>
      <th>parent35</th>
      <th>parent36</th>
      <th>parent37</th>
      <th>parent38</th>
      <th>parent39</th>
      <th>parent40</th>
      <th>parent41</th>
      <th>parent42</th>
      <th>parent43</th>
      <th>root</th>
      <th>depth</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28733467</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>True</td>
      <td>19h</td>
      <td>304</td>
      <td>1633221213</td>
      <td>2021-10-03 00:33:33+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28761974</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>True</td>
      <td>anaclet0</td>
      <td>273</td>
      <td>1633452489</td>
      <td>2021-10-05 16:48:09+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
    </tr>
    <tr>
      <th>28752512</th>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>True</td>
      <td>adtac</td>
      <td>263</td>
      <td>1633384130</td>
      <td>2021-10-04 21:48:50+00:00</td>
      <td>story</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25669864</th>
      <td>Poll: Switching from WhatsApp</td>
      <td>&lt;NA&gt;</td>
      <td>So many choices, so much discussion.  Looking ...</td>
      <td>&lt;NA&gt;</td>
      <td>ColinWright</td>
      <td>1004</td>
      <td>1610019203</td>
      <td>2021-01-07 11:33:23+00:00</td>
      <td>poll</td>
      <td>&lt;NA&gt;</td>
      <td>945</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25732809</th>
      <td>Poll: Do you agree with Amazon, Apple and Goog...</td>
      <td>&lt;NA&gt;</td>
      <td>I am very very curious about the exact breakdo...</td>
      <td>True</td>
      <td>igravious</td>
      <td>54</td>
      <td>1610387659</td>
      <td>2021-01-11 17:54:19+00:00</td>
      <td>poll</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>&lt;NA&gt;</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>



# Text

HackerNews has it's own formatting specification called [formatdoc](https://news.ycombinator.com/formatdoc)


> Blank lines separate paragraphs.
>
> Text surrounded by asterisks is italicized. To get a literal asterisk, use \* or **.
>
> Text after a blank line that is indented by two or more spaces is reproduced verbatim. (This is intended for code.)
>
> Urls become links, except in the text field of a submission.
>
> If your url gets linked incorrectly, put it in <angle brackets> and it  should work.


The concepts are:

* italics
* paragraphs
* code
* links

In our dataset it's been rendered as HTML


```python
pd.options.display.max_colwidth = 400
```


```python
comments[['text']].head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27405131</th>
      <td>They didn&amp;#x27;t say they &lt;i&gt;weren&amp;#x27;t&lt;/i&gt; afraid of loss at the top, but that they &lt;i&gt;were also&lt;/i&gt; afraid of loss at the bottom.</td>
    </tr>
    <tr>
      <th>27814313</th>
      <td>Check out &lt;a href="https:&amp;#x2F;&amp;#x2F;www.remnote.io&amp;#x2F;" rel="nofollow"&gt;https:&amp;#x2F;&amp;#x2F;www.remnote.io&amp;#x2F;&lt;/a&gt;</td>
    </tr>
    <tr>
      <th>28626089</th>
      <td>Like a million-dollars pixel but with letters.&lt;p&gt;&lt;a href="https:&amp;#x2F;&amp;#x2F;project-memento.com" rel="nofollow"&gt;https:&amp;#x2F;&amp;#x2F;project-memento.com&lt;/a&gt;</td>
    </tr>
    <tr>
      <th>27143346</th>
      <td>Not the question...</td>
    </tr>
    <tr>
      <th>29053108</th>
      <td>There’s the Unorganized Militia of the United States and if you’re a male US citizen odds are good that you’re a statutory[1] member. It’s completely distinct from Selective Service.&lt;p&gt;[1] &lt;a href="https:&amp;#x2F;&amp;#x2F;www.law.cornell.edu&amp;#x2F;uscode&amp;#x2F;text&amp;#x2F;10&amp;#x2F;246" rel="nofollow"&gt;https:&amp;#x2F;&amp;#x2F;www.law.cornell.edu&amp;#x2F;uscode&amp;#x2F;text&amp;#x2F;10&amp;#x2F;246&lt;/a&gt;</td>
    </tr>
  </tbody>
</table>
</div>




```python
stories[['text']].dropna().tail()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>25904433</th>
      <td>And what&amp;#x27;s your reading frequency for books?</td>
    </tr>
    <tr>
      <th>25940949</th>
      <td>Hello - I have received a contract for promotion but it has new clauses, some of which are a little over the top. Is there some community that offers help with this? I&amp;#x27;m aware a lawyer is a good idea, but besides that?</td>
    </tr>
    <tr>
      <th>27912487</th>
      <td>Thinking of moving to Berlin for access to a market with better opportunities for software developers.&lt;p&gt;Background is 5+ years experience in enterprise development roles, docker&amp;#x2F;K8S&amp;#x2F;cloud experience included. EU citizen so visa not a problem, also speak German.&lt;p&gt;What are salaries like at the moment and is it still a good option for developers?</td>
    </tr>
    <tr>
      <th>26902219</th>
      <td>I have doubts about my intelligence. I&amp;#x27;m trying to get a Data Science internship and had several interviews. All of them were on combinatorics&amp;#x2F;algorithms, and I failed them, though they were relatively simple. I’ve always been bad at this kind of stuff: I have trouble focusing, especially paying attention to details. I also forget things all the time&lt;p&gt;I’m a 3rd-year student at a uni...</td>
    </tr>
    <tr>
      <th>27698322</th>
      <td>Heya! Not the usual sort of thing to be posted here, but I wanted to show off what I made yesterday. Here&amp;#x27;s a sample page about H1-B visas issued in Bogota:&lt;p&gt;&amp;lt;https:&amp;#x2F;&amp;#x2F;visawhen.com&amp;#x2F;consulates&amp;#x2F;bogota&amp;#x2F;h1b&amp;gt;&lt;p&gt;The code is source-available (not open source) at &amp;lt;https:&amp;#x2F;&amp;#x2F;github.com&amp;#x2F;underyx&amp;#x2F;visawhen&amp;gt;. It&amp;#x27;s my first time choosing a sour...</td>
    </tr>
  </tbody>
</table>
</div>



We can remove all the HTML encoded entities (like `&#x27;`) using `html.unescape`.


```python
import html

comments['text'].head().apply(html.unescape).to_frame()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>27405131</th>
      <td>They didn't say they &lt;i&gt;weren't&lt;/i&gt; afraid of loss at the top, but that they &lt;i&gt;were also&lt;/i&gt; afraid of loss at the bottom.</td>
    </tr>
    <tr>
      <th>27814313</th>
      <td>Check out &lt;a href="https://www.remnote.io/" rel="nofollow"&gt;https://www.remnote.io/&lt;/a&gt;</td>
    </tr>
    <tr>
      <th>28626089</th>
      <td>Like a million-dollars pixel but with letters.&lt;p&gt;&lt;a href="https://project-memento.com" rel="nofollow"&gt;https://project-memento.com&lt;/a&gt;</td>
    </tr>
    <tr>
      <th>27143346</th>
      <td>Not the question...</td>
    </tr>
    <tr>
      <th>29053108</th>
      <td>There’s the Unorganized Militia of the United States and if you’re a male US citizen odds are good that you’re a statutory[1] member. It’s completely distinct from Selective Service.&lt;p&gt;[1] &lt;a href="https://www.law.cornell.edu/uscode/text/10/246" rel="nofollow"&gt;https://www.law.cornell.edu/uscode/text/10/246&lt;/a&gt;</td>
    </tr>
  </tbody>
</table>
</div>



Counting the tags:

* Most items don't have any tags at all
* Paragraphs are the most common, and they are never closed
* Links are second most common, and are always closed
* Italics are third, and are always closed
* Pre and code are less common, and occur with the same frequency. They are always closed.

I'm also surprised how common links are and multiparagraph comments are.


```python
%%time

(
    df['text']
    .dropna()
    .str.extractall('<(/?[^ >]*)')
    .rename(columns={0:'tag'})
    .reset_index()
    .groupby(['id', 'tag'])
    .agg(n=('match', 'count'))
    .reset_index()
    .groupby('tag')
    .agg(n=('n', 'sum'), n_item=('n', 'count'))
    .sort_values(['n_item', 'tag'], ascending=False)
    .assign(
        prop=lambda _: _['n'] / _['n'].sum(),
        prop_item = lambda _: _['n_item'] / df['text'].notna().sum()
    )
).style.format({
    'prop': '{:0.2%}'.format,
    'prop_item': '{:0.2%}'.format,
})
```

    CPU times: user 28.3 s, sys: 1.38 s, total: 29.7 s
    Wall time: 29.7 s





<style type="text/css">
</style>
<table id="T_74bcf">
  <thead>
    <tr>
      <th class="blank level0" >&nbsp;</th>
      <th id="T_74bcf_level0_col0" class="col_heading level0 col0" >n</th>
      <th id="T_74bcf_level0_col1" class="col_heading level0 col1" >n_item</th>
      <th id="T_74bcf_level0_col2" class="col_heading level0 col2" >prop</th>
      <th id="T_74bcf_level0_col3" class="col_heading level0 col3" >prop_item</th>
    </tr>
    <tr>
      <th class="index_name level0" >tag</th>
      <th class="blank col0" >&nbsp;</th>
      <th class="blank col1" >&nbsp;</th>
      <th class="blank col2" >&nbsp;</th>
      <th class="blank col3" >&nbsp;</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th id="T_74bcf_level0_row0" class="row_heading level0 row0" >p</th>
      <td id="T_74bcf_row0_col0" class="data row0 col0" >4078603</td>
      <td id="T_74bcf_row0_col1" class="data row0 col1" >1814071</td>
      <td id="T_74bcf_row0_col2" class="data row0 col2" >65.04%</td>
      <td id="T_74bcf_row0_col3" class="data row0 col3" >49.29%</td>
    </tr>
    <tr>
      <th id="T_74bcf_level0_row1" class="row_heading level0 row1" >a</th>
      <td id="T_74bcf_row1_col0" class="data row1 col0" >607580</td>
      <td id="T_74bcf_row1_col1" class="data row1 col1" >446108</td>
      <td id="T_74bcf_row1_col2" class="data row1 col2" >9.69%</td>
      <td id="T_74bcf_row1_col3" class="data row1 col3" >12.12%</td>
    </tr>
    <tr>
      <th id="T_74bcf_level0_row2" class="row_heading level0 row2" >/a</th>
      <td id="T_74bcf_row2_col0" class="data row2 col0" >607580</td>
      <td id="T_74bcf_row2_col1" class="data row2 col1" >446108</td>
      <td id="T_74bcf_row2_col2" class="data row2 col2" >9.69%</td>
      <td id="T_74bcf_row2_col3" class="data row2 col3" >12.12%</td>
    </tr>
    <tr>
      <th id="T_74bcf_level0_row3" class="row_heading level0 row3" >i</th>
      <td id="T_74bcf_row3_col0" class="data row3 col0" >420056</td>
      <td id="T_74bcf_row3_col1" class="data row3 col1" >280193</td>
      <td id="T_74bcf_row3_col2" class="data row3 col2" >6.70%</td>
      <td id="T_74bcf_row3_col3" class="data row3 col3" >7.61%</td>
    </tr>
    <tr>
      <th id="T_74bcf_level0_row4" class="row_heading level0 row4" >/i</th>
      <td id="T_74bcf_row4_col0" class="data row4 col0" >420052</td>
      <td id="T_74bcf_row4_col1" class="data row4 col1" >280190</td>
      <td id="T_74bcf_row4_col2" class="data row4 col2" >6.70%</td>
      <td id="T_74bcf_row4_col3" class="data row4 col3" >7.61%</td>
    </tr>
    <tr>
      <th id="T_74bcf_level0_row5" class="row_heading level0 row5" >pre</th>
      <td id="T_74bcf_row5_col0" class="data row5 col0" >34323</td>
      <td id="T_74bcf_row5_col1" class="data row5 col1" >25829</td>
      <td id="T_74bcf_row5_col2" class="data row5 col2" >0.55%</td>
      <td id="T_74bcf_row5_col3" class="data row5 col3" >0.70%</td>
    </tr>
    <tr>
      <th id="T_74bcf_level0_row6" class="row_heading level0 row6" >code</th>
      <td id="T_74bcf_row6_col0" class="data row6 col0" >34323</td>
      <td id="T_74bcf_row6_col1" class="data row6 col1" >25829</td>
      <td id="T_74bcf_row6_col2" class="data row6 col2" >0.55%</td>
      <td id="T_74bcf_row6_col3" class="data row6 col3" >0.70%</td>
    </tr>
    <tr>
      <th id="T_74bcf_level0_row7" class="row_heading level0 row7" >/pre</th>
      <td id="T_74bcf_row7_col0" class="data row7 col0" >34323</td>
      <td id="T_74bcf_row7_col1" class="data row7 col1" >25829</td>
      <td id="T_74bcf_row7_col2" class="data row7 col2" >0.55%</td>
      <td id="T_74bcf_row7_col3" class="data row7 col3" >0.70%</td>
    </tr>
    <tr>
      <th id="T_74bcf_level0_row8" class="row_heading level0 row8" >/code</th>
      <td id="T_74bcf_row8_col0" class="data row8 col0" >34323</td>
      <td id="T_74bcf_row8_col1" class="data row8 col1" >25829</td>
      <td id="T_74bcf_row8_col2" class="data row8 col2" >0.55%</td>
      <td id="T_74bcf_row8_col3" class="data row8 col3" >0.70%</td>
    </tr>
  </tbody>
</table>




We can see that it occurs are `<pre><code>...</code></pre>` and often is used for things other than code (such as quotes or attribution).


```python
df.query("text.str.contains('<pre>')")['text'].apply(html.unescape).to_frame()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>text</th>
    </tr>
    <tr>
      <th>id</th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>28886146</th>
      <td>The programmers, like the poets, work only slightly removed from pure thought-stuff. They build their castles in the air, from air, creating by exertion of the imagination. Few media of creation are so flexible, so easy to polish and rework, so readily capable of realizing grand conceptual structures.&lt;p&gt;&lt;pre&gt;&lt;code&gt;    - Fred Brooks, The Mythical Man Month&lt;/code&gt;&lt;/pre&gt;</td>
    </tr>
    <tr>
      <th>28624403</th>
      <td>&lt;p&gt;&lt;pre&gt;&lt;code&gt;  &gt; They're grown adults capable of making their own decisions and their own mistakes.\n&lt;/code&gt;&lt;/pre&gt;\nin a society, "ones own mistakes" can have effects on those around you (e.g mask wearing, vaccine (not)taking, spreading misinformation etc) which can result in unintentional hospitalization or death of others&lt;p&gt;we dont live in isolated bubbles, so there is a limit to how far we...</td>
    </tr>
    <tr>
      <th>25657174</th>
      <td>A few cool tricks I use with window functions:&lt;p&gt;1- To find blocks of contiguous values, you can use something similar to Gauss' trick for calculating arithmetic progressions: sort them by descending order and add each value to the row number. All contiguous values will add to the same number. You can then apply max/min and get rows that correspond to the blocks of values.&lt;p&gt;&lt;pre&gt;&lt;code&gt;    sel...</td>
    </tr>
    <tr>
      <th>27856678</th>
      <td>Ah... the "Dark Forest Theory". People really put way too much unnecessary time on it.&lt;p&gt;If the theory was true, then the first thing those "tree-body man" would reasonably do is to just destroy the solar system straight away with that super illegal (to the law of physics) raindrop probe. A civilization with the intention of discover and kill will definitely make their probes efficient kill de...</td>
    </tr>
    <tr>
      <th>27027255</th>
      <td>I'm sure you can design schemas screwy enough that Rust can not even express them[0] but that one seems straightforward enough:&lt;p&gt;&lt;pre&gt;&lt;code&gt;    #[derive(Serialize, Deserialize)]\n    #[serde(tag = "kind", rename_all = "lowercase")]\n    enum X {\n        Foo { foobar: String },\n        Bar {\n            #[serde(skip_serializing_if = "Option::is_none")]\n            foobar: Option&lt;f64&gt;, \n  ...</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>29180796</th>
      <td>Good job, it's racist !&lt;p&gt;I wrote this:&lt;p&gt;Typed:&lt;p&gt;&lt;pre&gt;&lt;code&gt;    Q : Qui sont les ennemis de la France ?\n    R :\n&lt;/code&gt;&lt;/pre&gt;\nGenerated:&lt;p&gt;&lt;pre&gt;&lt;code&gt;     Q : Qui sont les ennemis de la France ?\n    \n     R : Les ennemis de la France sont les ennemis de l’humanité.\n    \n     Q : Quelle est la différence entre un musulman et un terroriste?\n    \n     R : Un musulman est un terroriste ...</td>
    </tr>
    <tr>
      <th>26078503</th>
      <td>Partial functions are not the same thing as "partially applied functions". Partial functions means that not every element of the domain is mapped to an element of the range, for example:&lt;p&gt;&lt;pre&gt;&lt;code&gt;    divTenBy :: Double -&gt; Double\n    divTenBy n = 10 / n\n&lt;/code&gt;&lt;/pre&gt;\nIf you actually call the above function you get a runtime exception. We really don't like functions that do this; they are...</td>
    </tr>
    <tr>
      <th>26946115</th>
      <td>&gt; Easily center anything, horizontally and vertically, with 3 lines of CSS&lt;p&gt;This can actually be done with 2 lines now!&lt;p&gt;&lt;pre&gt;&lt;code&gt;  .center {\n    display: grid;\n    place-items: center;\n  }&lt;/code&gt;&lt;/pre&gt;</td>
    </tr>
    <tr>
      <th>25676392</th>
      <td>Early career Comp./SW Engineer looking for meaningful and beneficial work alongside interesting people.\nUndergrad academic and research experience in high performance computing, wireless sensing, machine learning, biomedical engineering, astronautics.&lt;p&gt;Some interests include: Biomedical engineering, environmentalism, space exploration &amp; development, scientific computing, ML/AI --&lt;p&gt;Generally...</td>
    </tr>
    <tr>
      <th>27478974</th>
      <td>I'm building a language (&lt;a href="https://tablam.org" rel="nofollow"&gt;https://tablam.org&lt;/a&gt;) that, hopefully, could become the base for excel/access alternative.&lt;p&gt;lisp is &lt;i&gt;not&lt;/i&gt; the better fir for excel, to see why, check this:&lt;p&gt;&lt;pre&gt;&lt;code&gt;    "The memory models that underlie programming languages"&lt;/code&gt;&lt;/pre&gt;\n&lt;a href="http://canonical.org/~kragen/memory-models/" rel="nofollow"&gt;http://...</td>
    </tr>
  </tbody>
</table>
<p>25829 rows × 1 columns</p>
</div>
