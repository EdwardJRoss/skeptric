---
categories:
- python
- html
date: '2020-08-24T21:27:32+10:00'
image: /images/html2plain2.png
title: Python HTML Parser
---

A lot of information is embedded in HTML pages, which contain both human text and markup.
If you ever want to extract this information, [don't use regex](https://stackoverflow.com/a/1732454/) use a parser.
Python has an inbuilt library [`html.parser`](https://docs.python.org/3/library/html.parser.html) library to do just that.

The excellent [html2text](https://github.com/Alir3z4/html2text) library uses it to parse HTML into markdown, which you can use for [removing formatting](/html-to-text).
However for your own purposes you can use a similar approach to build a custom parser by subclassing `HTMLParser`.

Here's a simple example of a parser that tries to convert HTML to plain text.
You would use it like this:

```python
converter = HTMLTextConverter()
plain_text = converter('<html><h1>Example</h1><p>Hello world!</p></html>')
plain_text == 'Example\nHello world!'
```

When you feed HTML to a HTMLParser it executed `handle_starttag` whenever it encounters a new open tag, `handle_endtag` whenever it encounters a new close tag, and `handle_data` whenever it encounters data between tags.

To insert newlines whenever we hit a block level tag we can implement a custom `handle_starttag`, that adds a newline to an `output` method.

```python
    def handle_starttag(self, tag, attrs):
        if tag in BLOCK_TAGS:
            self.output('\n')
        elif tag in INLINE_TAGS:
            pass
        else:
            raise ValueError('Unexpected tag %s', tag)
```

In this case we don't need to do anything special with endtags, but we do need to output all data.
We will strip off newlines, because they won't be shown in HTML output.

```python
    def handle_data(self, data):
        self.output(data.strip('\n'))
```

The output method is one we need to add ourselves; we can append the output to internal state in a list called `outdata`.
We add to a list rather than append to a string because Python strings are immutable which means we'd need to create a whole new string object when we append a single character which is very inefficient if the string gets large.

```python
    def output(self, data):
        self.outdata.append(data)
```

Of course we need to initialise `self.outdata` to an empty list.

```python
    def __init__(self) -> None:
        self.outdata = []
        super().__init__()
```

Finally we can provide a nice interface that does all the work when we call `converter(html)` by implementing the `__call__` magic method.

```python
    def __call__(self, html):
        self.feed(html)
        output = ''.join(self.outdata).strip()
        self.reset()
        return output
```

That's all there is to implementing a simple HTML transformation in Python.
If you wanted more complex transformations you would need to track more pieces of state; the html2text code is a good example of how this can work.

## Full example listing

Here's an example listing of the HTML Parser.
The functionality is very basic; it's likely to produce way too much whitespace in certain cases, and fail on many HTML documents.
However it's a reasonable starting point for building a customer HTML transformation function.

```python
BLOCK_TAGS = (
  'html', 'p', 'br',
  'li', 'ul', 'ol',
  'blockquote',
  'table', 'tbody', 'tr',
  )
INLINE_TAGS = (
  'strong', 'ul', 'em', 'i', 'b',
  'a', 'figure', 'img',
  'h1', 'h2', 'h3', 'h4', 'h5', 'h6',
  'td',
  )

from html.parser import HTMLParser
class HTMLTextConverter(HTMLParser):
    def __init__(self) -> None:
        self.outdata = []
        super().__init__()

    def __call__(self, html):
        self.feed(html)
        output = ''.join(self.outdata).strip()
        self.reset()
        return output

    def reset(self):
        super().reset()
        self.outdata = []

    def output(self, data):
        self.outdata.append(data)

    def handle_starttag(self, tag, attrs):
        if tag in BLOCK_TAGS:
            self.output('\n')
        elif tag in INLINE_TAGS:
            pass
        else:
            raise ValueError('Unexpected tag %s', tag)

    def handle_endtag(self, tag):
        pass

    def handle_data(self, data):
        self.output(data.strip('\n'))
```