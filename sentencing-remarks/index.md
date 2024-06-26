---
categories:
- python
date: '2021-07-09T22:15:35+10:00'
image: /images/sentencing_remarks.png
title: Getting Sentencing Data
---

In [*Noise*](https://readnoise.com/) by Kahneman, Sibony and Sunstein, they discuss how much variation there is in sentences for very similar circumstances.
This made me curious how much variation in sentencing there was in the Australian legal systems.
In Victoria the sentencing council [publishes statistics on sentences by offence](https://www.sentencingcouncil.vic.gov.au/sacstat/home.html) that helps answer this question.
But I'd recently seen some news about South Australia, and was wondering if I could get the information for South Australia.

The Courts Administration Authority of South Australia published [Sentencing Remarks](http://www.courts.sa.gov.au/SentencingRemarks/Pages/default.aspx) for judgements, but it only keeps data back one month.
Sentencing remarks are very informative because they explain the rationale behind the sentencing; the factors the judge consciously weighted (in *Noise* they talk about unconscious factors citing that [Judges give more favourable rulings after lunch](https://www.pnas.org/content/108/17/6889?ijkey=02cd11a80bb706a8a8296e760f84b0b08ba32b3b&keytype2=tf_ipsecsha), but this could just be [correlation with whether the defendants have legal representation](https://www.pnas.org/content/108/42/E833.full)).
The [Australasian Legal Information Institute](http://www.austlii.edu.au/) has a great public database of case law, but it doesn't have all the sentencing remarks for South Australia (there were cases where I could only find the judgements, not the sentencing remarks).

It's fantastic that the South Australian Sentencing Remarks are publicly available online; they inform the public on how these decisions are made which should be in line with community expectations.
Without that feedback loop it would be harder for the public to find out why a sentence was given.
However it's a pity it only goes back one month, the only way I could find to get sentences beyond that was by going through news media websites.
It would be a great service if someone archived these and made them publicly available.

For a very short time in 2013 the Wayback Machine captured some of these sentencing remarks.
The remarks are listed on `http://www.courts.sa.gov.au/SentencingRemarks/Pages/default.aspx` and link to URLs like `http://www.courts.sa.gov.au/SentencingRemarks/Pages/lightbox.aspx?IsDlg=1&Filter=NNNN` where the last characters are some number unique to the case.

While these are easy for a human to read, I thought it could be useful to try to extract relevant sections of the remarks programatically using Python.
Here's a brief snippet to read an archived sentencing remark into Python using [CDX Toolkit](https://github.com/cocrawler/cdx_toolkit/) (which can also [access Common Crawl](/searching-100b-pages-cdx)):

```python
import cdx_toolkit

# Fetcher for Internet Archive's Wayback Machine
cdx = cdx_toolkit.CDXFetcher(source='ia')

# Search for results
url = 'http://www.courts.sa.gov.au/SentencingRemarks/Pages/lightbox.aspx*'
# At running gets 784 results, 759 of them unique
results = [result for result in cdx.get(url) if 'Filter=' in result.data['url']]

# Get the plain text
text = extract_text(results[0].text)
```

The `extract_text` uses a very simple approach of converting the HTML to text (there are [more robust ways to do this](/html-to-text)):

```python
import parsel
import re

def extract_text(html: str) -> str:
    sel = parsel.Selector(r.text)
    selector = sel.css('div.item *::text')
    text = ''.join([remove_multiple_newlines(fragment) for fragment in selector.getall()])
    return remove_multiple_spaces(text)

def remove_multiple_spaces(text: str) -> str:
    return re.sub('[ \xa0]+', ' ', text)

def remove_multiple_newlines(text: str) -> str:
    # Insert a space because these can happen across markup
    return re.sub(r'\r\n(\r\n)?', r' \1', text)
```

From here you could use text matching or NLP to find relevant parts of the remarks.
