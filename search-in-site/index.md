---
categories:
- general
date: '2020-08-01T08:00:00+10:00'
image: /images/skeptric_schema_search.png
title: Searching within a Website
---

Some websites, like this one, have a lot of content but have no search function.
Others have search but it performs poorly, for example Bunnings has great category pages but the search never hits it.
Fortunately there's a simple way to search these sites with the `site:` search operator.

If I want to search for articles about jobs just in this website I can type: `site:skeptric.com job` into either Google or Bing.
I find this really handy because I have over 150 articles and I often forget what I wrote.
For this website I could also search with Github (e.g. [`repo:EdwardJRoss/skeptric job`](https://github.com/EdwardJRoss/skeptric/search?q=repo%3AEdwardJRoss%2Fskeptric+job&unscoped_q=repo%3AEdwardJRoss%2Fskeptric+job)), or just download and search with grep; but that's generally less convenient.

Another useful example is the Australian Beaurau of Statistics.
They run lots of useful surveys, but it's really hard to find them through navigating the website.
However if I search for [`site:abs.gov.au motor vehicle`](https://www.bing.com/search?q=site%3Aabs.gov.au+motor+vehicle) I can immediately find survey and census data on Motor Vehicles.

There are many other useful advanced search operators, like to get results with a specific filetype, language, location or words in the title.
Google has a [partial list](https://support.google.com/websearch/answer/2466433?hl=en) but you can access much more through their [advanced search](https://www.google.com/advanced_search).
Bing has more [comprehensive documentation](https://help.bing.microsoft.com/#apex/18/en-US/10001/-1) about its search operators.

However I've found they don't always function as I would expect; for example if I want to search for Jupyter notebooks on RoBERTa I would try a search like `roberta ext:ipynb`.
This works perfectly in Google, but in Bing returns HTML articles about people named Roberta.
It's useful to try both when you're looking for something very specific.
If you want to do a broad shallow search of what's in the internet or do some analysis, then the Common Crawl [columnar index](/common-crawl-index-athena) is a better tool.