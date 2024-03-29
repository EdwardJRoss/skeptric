---
categories:
- python
- html
- data
- nlp
date: '2022-06-19T11:03:34+10:00'
image: /images/sourcemap_html_tags.png
title: Source Map HTML Tags in Python
---

When using [NLP in HTML](/html-nlp) it's useful to extract the HTML tags which change the [meaning of the text](/meaning-in-html).
I've already shown how to [source map](/source-mapping-html) the text of HTML using Python's inbuild HTMLParser.
We'll now adapt this to the task of getting the tags, which could be considered as a sort of annotation on the text.

I'm actually surprised there's not good tooling for doing this in Python.
It seems like the fast HTML parsers like lxml, [html5-parser](https://html5-parser.readthedocs.io/en/latest/) and [Selectolax](https://github.com/rushter/selectolax) don't make the source positions accessible through their API.
Python's inbuilt HTMLParser does, and it's [possible to get from html5lib](https://stackoverflow.com/a/30647679/6570019), and BeautifulSoup exposes them [via `sourceline` and `sourcepos`](https://www.crummy.com/software/BeautifulSoup/bs4/doc/#line-numbers).
Unfortunately with BeautifulSoup I can't find any way to get the *length* of the source tag; `len(str(node))` doesn't work because the parsing mutates the HTML (try parsing `<li>` and you'll see what I mean).
I think there still be useful ways to deconstruct the text using a parser, but let's see if we can source map the tags in a compatible way with the text.

We'll start with the same `MyHTMLParser` that we used to source map the text, but add an additional property `end_tag_index` to give the index of the end of a tag.
Recall the `current_index` will always be the index just before the tag starts, so searching for the next `>` is a reliable way to get the end of the tag.

```python
from html.parser import HTMLParser
from itertools import accumulate
import re

end_tag = re.compile('>')

class MyHTMLParser(HTMLParser):
    @property
    def end_tag_index(self):
        return end_tag.search(self.rawdata, self.current_index).end()

    def reset(self):
        super().reset()
        self.result = None

    @property
    def current_index(self):
        line, char = self.getpos()
        return self.line_lengths[line - 1] + char

    def __call__(self, data):
        self.reset()
        self.line_lengths = [0] + list(accumulate(len(line) for line in
                                                  data.splitlines(keepends=True)))
        self.feed(data)
        self.close()
        return self.result
```

Then similar to how we handled text we can register the data at a start tag, and at an end tag append the end index and to a list.
However because tags can be arbitrarily nested we need a way to keep track of which tag we are closing.
One way to handle this is to use the deque collection as a FIFO ([First In First Out](https://en.wikipedia.org/wiki/FIFO_(computing_and_electronics))) queue; when we reach the end tag we close the most recent matching start tag.
There needs to a be a queue for every possible tag, so we use a `defaultdict` to track the queue for any possible tag.
Here's a sample implementation:


```python
from collections import deque, defaultdict

class HTMLTagExtract(MyHTMLParser):
    def reset(self):
        super().reset()
        self.tags = defaultdict(deque)
        self.result = []

    def handle_starttag(self, tag, attrs):
        self.tags[tag].append({'tag': tag,
                               'attrs': attrs,
                               'start': self.current_index,
                               'start_inside': self.end_tag_index})

    def pop_tag(self, tag, end):
        # If there's no matching tag then recover
        if self.tags[tag]:
            tagdata = self.tags[tag].pop()
            tagdata['end'] = end
            tagdata['end_inside'] = self.current_index
            self.result.append(tagdata)

    def handle_endtag(self, tag):
        self.pop_tag(tag, self.end_tag_index)

    def close(self):
        super().close()
        for tag in self.tags:
            while self.tags[tag]:
                self.pop_tag(tag, self.current_index)
```

And here's an example of using it:

```python
parser = HTMLTagExtract()

html = '<p>Hello <i>world</ i ><p><ul><li>I am here<li></ul>I am well</p>'

for tag in parser(html):
    print('%4s %3d %3d %s' % (tag['tag'], tag['start'], tag['end'], html[tag['start']:tag['end']]))
```

which outputs:

```
   i   9  23 <i>world</ i >
  ul  26  52 <ul><li>I am here<li></ul>
   p  23  65 <p><ul><li>I am here<li></ul>I am well</p>
   p   0  65 <p>Hello <i>world</ i ><p><ul><li>I am here<li></ul>I am well</p>
  li  43  65 <li></ul>I am well</p>
  li  30  65 <li>I am here<li></ul>I am well</p>
```

If you look carefully you'll notice the last `p` ranges from 0 to 65, and in particular it has another `p` inside it.
This isn't the correct thing to do either in the [whatwg standard](https://html.spec.whatwg.org/multipage/grouping-content.html#the-p-element) or in practice; the second `p` should end the first one.
Python's HTML Parser isn't a strict parser in this sense.
I went through the standard and searched for `end tag can be omitted` to find these implicit closing tags, which can happen either on opening of certain tags, or on the closing of a parent tag.
I input them as dictionaries as follows:

```python
TAG_CLOSED_BY_CLOSE = {
    'p': ['article', 'aside', 'nav', 'section'],
    'li': ['ol', 'ul', 'menu'],
    'dd': ['dl', 'div'],
    'rt': ['ruby'],
    'rp': ['ruby'],
    'caption': ['table'],
    'colgroup': ['table'],
    'tbody': ['table'],
    'tr': ['table', 'tbody', 'tfoot', 'thead'],
    'td': ['tr'],
    'th': ['tr'],
}

TAG_CLOSED_BY_OPEN = {
    'p': ['address', 'article', 'aside', 'blockquote',
          'details', 'div', 'dl', 'fieldset', 'figcaption',
          'figure', 'footer', 'form',
          'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'header',
          'hgroup', 'hr', 'main', 'menu', 'nav', 'ol', 'p',
          'pre', 'section', 'table', 'ul'],
    'li': ['li'],
    'dt': ['dt', 'dd'],
    'dd': ['dt', 'dd'],
    'rt': ['rp', 'rt'],
    'rp': ['rp', 'rt'],
    'tbody': ['tbody', 'tfoot'],
    'thead': ['tbody', 'tfoot'],
    'tr': ['tr'],
    'td': ['td', 'th'],
    'th': ['td', 'th'],
}
```

Let's also wrap the tag structures as dataclasses to make them easier to use.
We have a `TagSpan` that captures the location, the tag and attributes, as well as a reference to the parent document that contains the HTML.

```python
from __future__ import annotations
from dataclasses import dataclass
from typing import optional

@dataclass
class TagSpan:
    start: int
    start_inside: int
    end: int
    end_inside: int
    tag: Optional[str]
    attrs: dict[tuple[str, str]]
    doc: 'HTMLSpanDoc'

    @property
    def html(self):
        return self.doc.html[self.start:self.end]
```

The parent document contains all the `TagSpan`s, that can be appended to or iterated over, as well as the source html.

```python
@dataclass
class HTMLSpanDoc:
    tag_spans: list[TagSpan] = field(default_factory=list)
    html: str = ''

    def append(self, item: TagSpan) -> None:
        self.tag_spans.append(item)

    def __iter__(self):
        return iter(self.tag_spans)
```


Now we have all the ingredients we can extend our previous parser to use them:

```python
class HTMLTagExtract(MyHTMLParser):
    def __init__(self,
                 tag_closed_by_open=TAG_CLOSED_BY_OPEN,
                 tag_closed_by_close=TAG_CLOSED_BY_CLOSE):
        super().__init__()
        self.tag_closed_by_open = tag_closed_by_open
        self.tag_closed_by_close = tag_closed_by_close

    def feed(self, data):
        super().feed(data)
        self.result.html += data

    def reset(self):
        super().reset()
        self.tags = defaultdict(deque)
        self.result = HTMLSpanDoc()

    def omitted_tag_close(self, tag, close_map):
        for current_tag, parent_tags in close_map.items():
            if tag in parent_tags and self.tags[current_tag]:
                self.pop_tag(current_tag, self.current_index)
                self.omitted_tag_close(current_tag, self.tag_closed_by_close)

    def handle_starttag(self, tag, attrs):
        self.omitted_tag_close(tag, self.tag_closed_by_open)
        self.tags[tag].append({'tag': tag,
                               'start': self.current_index,
                               'start_inside': self.end_tag_index,
                               'attrs': attrs,
                               'doc': self.result})

    def pop_tag(self, tag, end):
        if self.tags[tag]:
            self.result.append(
                TagSpan(**self.tags[tag].pop(),
                        end=end,
                        end_inside=self.current_index)
            )

    def handle_endtag(self, tag):
        self.omitted_tag_close(tag, self.tag_closed_by_close)
        self.pop_tag(tag, self.end_tag_index)

    def close(self):
        super().close()
        for tag in self.tags:
            while self.tags[tag]:
                self.pop_tag(tag, self.current_index)
```

We can run our extraction again, iterating over the `HTMLSpanDoc` and accessing the attributes of the `TagSpan`:

```python
parser = HTMLTagExtract()


html = '<p>Hello <i>world</ i ><p><ul><li>I am here<li></ul>I am well</p>'


for tag in parser(html):
    print('%4s %3d %3d %s' % (tag.tag, tag.start, tag.end, tag.html))
```

This time we get the paragraphs separated.
However notice the tag `I am well` doesn't occur inside any tag, this is because it's in an implicitly opened paragraph.

```
   i   9  23 <i>world</ i >
   p   0  23 <p>Hello <i>world</ i >
   p  23  26 <p>
  li  30  43 <li>I am here
  li  43  47 <li>
  ul  26  52 <ul><li>I am here<li></ul>
```

You could use this with the [HTMLTextExtractor for source mapping text](/source-mapping-html) to annotate tags like `<i>` by matching source indices, and to attribute the text to blocks (like paragraphs and list items).
Block attribution is actually a little tricky because we need to be sure that text always falls in exactly one block, and I don't think this is guaranteed by our block closing rules above (though we could enforce them).
Another more direct way to do this is to force only one block tag being active, like I tried in my [first post on Python HTML Parser](/python-html-parser).

The current implementation will really only be partly compatible with HTML5 (let alone HTML in the wild) and there are two distinct paths.
One is to use a realy HTML parser and do processing on this parse tree; for example `html5lib` could be used to generate valid HTML (closing all the tags) which could then be processed by the first version of our code.
A second option is to customize the parser for a particular use case; we won't always want things to the specification and we could specify our own block attribution.
