---
categories:
- python
- html
- data
- nlp
date: '2022-06-17T08:00:00+10:00'
image: /images/source_mapping_html.png
title: Source Mapping Text HTML in Python
---

Sometimes I want to extract text from HTML for processing, but I don't want to lose the context.
This is useful in [NLP with HTML](/html-nlp) because sometimes the context (is it emphasised, or in a header or list item) may be relevant for capturing information.
The common approaches of [converting HTML to text](/html-to-text) are lossy; you strip away the HTML and lose the context.
Instead we're going to do create a *source mapping* using the [Python HTML Parser](/python-html-parser), for each character of the output text we can find the corresponding character of the HTML that generated it.
This is similar to the non-destructive parsing approach SpaCy takes, and the idea of a [source map](https://firefox-source-docs.mozilla.org/devtools-user/debugger/how_to/use_a_source_map/index.html) for transformed code.

We're going to start by making our own subclass of [`HTMLParser`](https://docs.python.org/3/library/html.parser.html) with some useful functionality.
`HTMLParser` can give us the line and character number just before the data to be processed, we add a property `current_index` to get the current index.
We'll also make it so when you call it on `data` it gets parsed and whatever is in `self.result` gets returned.


```python
from html.parser import HTMLParser
from itertools import accumulate

class MyHTMLParser(HTMLParser):
    def reset(self):
        super().reset()
        self.result = None

    @property
    def current_index(self):
        line, char = self.getpos()
        return self.line_lengths[line - 1] + char

    def __call__(self, data):
        self.reset()
        self.line_lengths = [0] + list(accumulate(len(line) for line in data.splitlines(keepends=True)))
        self.feed(data)
        self.close()
        return self.result
```

Now we want to design a Text Extractor that extracts each span of text along with the start and end indices in the original HTML.
Something like this:

```python
parser = HTMLTextExtractor()

assert parser("Hello world") == [{'text': 'Hello world', 'start': 0, 'end': 11}]

assert parser("Hello <b>world</b>") == [{'text': 'Hello ', 'start': 0, 'end': 6},
                                        {'text': 'world', 'start': 9, 'end': 14}]
```

We can do this by capturing the data and index whenever we get text, and then when we hit the next tag capture the end index and append it to our result.

```python
class HTMLTextExtractor(MyHTMLParser):
    def reset(self):
        super().reset()
        self.result = []
        self.text = None

    def handle_data(self, data):
        self.text = {'text': data, 'start': self.current_index}

    def handle_starttag(self, tag, attrs):
        self.text_end()

    def handle_endtag(self, tag):
        self.text_end()

    def handle_comment(self, data):
        self.text_end()

    def text_end(self):
        if self.text:
            self.text['end'] = self.current_index
            self.result.append(self.text)
            self.text = ''

    def close(self):
        super().close()
        self.text_end()
```

This works but it's then difficult to do the source lookup.
We'll add some objects to make this easier.
Instead of directly capturing the text each text span can capture a `HTMLDoc` object containing the raw `html`.


```python
from dataclasses import dataclass, field
from __future__ import annotations

@dataclass
class HTMLTextSpan:
    start: int
    end: int
    doc: 'HTMLDoc'

    @property
    def text(self) -> str:
        return self.doc.html[self.start:self.end]
```

Our `HTMLDoc` in turn holds all the `text_spans` as well as the `html`.
We can `append` to our text_spans, find get the text, find the length of that text.

```python
@dataclass
class HTMLDoc:
    text_spans: list[HTMLTextSpan] = field(default_factory=list)
    html: str = ''

    def append(self, text: HTMLTextSpan) -> None:
        self.text_spans.append(text)

    @property
    def text(self) -> str:
        return ''.join(span.text for span in self.text_spans)

    def __len__(self) -> int:
        return sum((span.end - span.start) for span in self.text_spans)
```

For finding the source of a text index we just need to iterate through the text spans to that index, and extract the offset.

```python
class HTMLDoc
    ...
    def source_index(self, idx: int) -> int:
        if idx < 0:
            idx = len(self) + idx
        if not 0 <= idx < len(self):
            raise ValueError(f"Index {idx} not in range of {len(self)}")

        for span in self.text_spans:
            size = span.end - span.start
            if idx <= size:
                return span.start + idx
            idx -= size
        return span.end

```

Then for a non-HTML text we'd get the same thing:

```python
parser = HTMLTextExtractor()

assert parser("Hello world").text == 'Hello world'
assert parser("Hello world").source_index(0) == 0
assert parser("Hello world").source_index(6) == 6
assert parser("Hello world").source_index(10) == 10
```

With HTML we have to calculate the offset from tags:

```python
assert parser("Hello <b>world</b>").source_index(0) == 0
assert parser("Hello <b>world</b>").source_index(6) == 9
assert parser("Hello <b>world</b>").source_index(10) == 13
```

The key property of the source map is if we re-parse the source of a text we should get the same text back.

```python
html = "Hello <b>world</b>"

start_idx = 3
end_idx = 10

doc = parser(html)

html_source = html[doc.source_index(start_idx):doc.source_index(end_idx)]
assert parser(html_source).text == doc.text[start_idx:end_idx]
```

We can do this with pretty much the almost the same code as before:

```python
class HTMLTextExtractor(MyHTMLParser):
    def reset(self):
        super().reset()
        self.result = HTMLDoc()
        self.text_start = None

    def feed(self, data):
        super().feed(data)
        self.result.html += data


    def handle_starttag(self, tag, attrs):
        self.text_end()

    def handle_endtag(self, tag):
        self.text_end()

    def handle_data(self, data):
        self.text_start = self.current_index

    def text_end(self):
        if self.text_start is not None:
            text = HTMLTextSpan(start=self.text_start,
                                end=self.current_index,
                                doc=self.result)
            self.result.append(text)
            self.text_start = None

    def close(self):
        super().close()
        self.text_end()
```

That's all there is to a simple, bare bones HTML to text source map.
For real usecases you'd need to think about removing extra whitespace from the HTML, breaking across paragraphs, and extracting relevant HTML tags.
This could be then further processed using something like SpaCy and we could annotate the document with spans of relevant tags, such as emphasis.
