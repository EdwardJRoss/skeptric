---
categories:
- python
date: '2021-04-13T19:47:52+10:00'
image: /images/extract_links.png
title: Extracting Links From HTML
---

Sometimes you have a HTML webpage or [email](/python-imap) that you want to extract all the links from.
There's lots of ways to do this, but there's a simple solution in Python with BeautifulSoup:

```python
from bs4 import BeautifulSoup
def extract_links(html):
    soup = BeautifulSoup(html, 'html.parser')
    return [a.get('href') for a in soup.find_all('a') if a.get('href')]
```

Some other methods would be to use regular expressions (which would be faster than parsing, but a little harder to get right), directly going through a parse tree or using lxml.
These other solutions would likely be a bit faster, but I like the flexibility of BeautifulSoup (especially with it's `select` method for CSS selectors).