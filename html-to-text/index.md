---
categories:
- python
- nlp
date: '2020-08-08T08:00:00+10:00'
image: /images/html2plain.png
title: Converting HTML to Text
---

I've been thinking about how to convert [HTML to Text for NLP](/html-nlp).
We want to at least extract the text, but if we can preserve some of the formatting it can make it easier to extract information down the line.
Unfortunately it's a little tricky to get the segmentation right.

The standard [answers on Stack Overflow](https://stackoverflow.com/questions/14694482/converting-html-to-text-with-python) are to use [Beautiful Soup's](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) `getText` method.
Unfortunately this just turns *every* tag into the argument, whether it is block level or inline.
This means for a lot of compact HTML it changes the meaning.

The pragmatic answer I've ended up with is to convert the text to Markdown with `html2text`, parse the Markdown back into HTML, and then converting that HTML to text.
This is ridiculously inefficient, but lets me offload the processing logic to external tools and does a good enough job.

The final solution looks like this:

```python
def html2plain(html):
    md = html2md(html)
    md = normalise_markdown_lists(md)
    html_simple = mistletoe.markdown(md)
    text = BeautifulSoup(html_simple).getText()
    text = fixup_markdown_formatting(text)
    return text
```

# The problem

For example the following HTML document:

```html
<b>Section</b><br />A list<ul><li>Item <b>1</b></li>
```

Would be converted to something where we lose all sentence and section structure:

```
Section A list Item 1
```

We can convert the tags into newlines with BeautifulSoup but that will break across inline tags:

```
Section
A list
Item
1
```

The best option is to write your own HTML parser, but that's hard because you have to decide what to do with every case and deal with the complexities of real HTML.
Another way is to first convert it to Markdown with [html2text](https://github.com/Alir3z4/html2text).
Then we would get something we may be able to parse:

```
Section

A list

* Item *1*
```

We can then convert that *back* into simple HTML or plain text.

```python
from bs4 import BeautifulSoup
from mistletoe import markdown
from html2text import HTML2Text

md = HTML2Text().handle(html)
html2 = markdown(md)
text = BeautifulSoup(html2).getText()
```

# Converting the HTML to Markdown

The html2text library does a good job of converting HTML to markdown.
We need to give it a little configuration to get the output we want.
In particular to turn off line wrapping we need to set the `body_width` to 0.
I also ignore anchors and images since they are rare and I have no way of dealing with them.


```python
def html2md(html):
    parser = HTML2Text()
    parser.ignore_images = True
    parser.ignore_anchors = True
    parser.body_width = 0
    md = parser.handle(html)
    return md
```

# Normalising Lists

HTML has a standard way of creating lists; `<ul>` and `<li>` tags.
However surprisingly often I find custom lists with formats like `List<br />- Item 1`.
We can convert these kinds of lists to look the same as a Markdown list with a little bit of regex:

```python
def normalise_markdown_lists(md):
    return re.sub(r'(^|\n) ? ? ?\\?[·--*]( \w)', r'\1  *\2', md)
```

# Converting the Markdown back to Text

There are a bunch of Markdown parsers, but [mistletoe](https://github.com/miyuchina/mistletoe) seems to be a good one.
The main benefit of going through Markdown is irrelevant tags are stripped off, and the mistletoe HTML output is consistently formatted.
In particular there are line breaks around block level formats, which may not be true for the source HTML.

```python
html_simple = mistletoe.markdown(md)
text = BeautifulSoup(html_simple).getText()
```

# Cleaning up processing errors

As nice as html2text is, it has issues with [multiple kinds of emphasis](/html2text_bi) and [repeated emphasis](/html2text-doubleemph).
For the repeated emphasis I remove any left over double stars.
Sometimes tables seem to leave an extra vertical strut, `|`.
I also clean up final whitespace.

```python
def fixup_markdown_formatting(text):
    # Strip off table formatting
    text = re.sub(r'(^|\n)\|\s*', r'\1', text)
    # Strip off extra emphasis
    text = re.sub(r'\*\*', '', text)
    # Remove trailing whitespace and leading newlines
    text = re.sub(r' *$', '', text)
    text = re.sub(r'\n\n+', r'\n\n', text)
    text = re.sub(r'^\n+', '', text)
    return text
```

# Testing it out

This pipeline was actually developed by trialing it on [some example job ads](https://github.com/EdwardJRoss/job-advert-analysis/blob/cc_pipeline/notebooks/Converting%20HTML%20to%20Text.ipynb).
The next step would be to create some formal tests based on these examples, but I'm happy to start with this until there are enough issues to improve it.