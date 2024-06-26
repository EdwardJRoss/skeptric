---
categories:
- python
date: '2020-08-05T08:00:00+10:00'
image: /images/html2text_error.png
title: An edge bug in html2text
---

I've been trying to find a way of converting HTML to something meaningful for NLP.
The [html2text](https://github.com/Alir3z4/html2text) library converts HTML to markdown, which strips away a lot of the meaningless markup.
But I quickly hit an edge case where it fails, because parsing HTML is surprisingly difficult.

I was parsing some HTML that looked like this:

```
Some text.<br /><i><b>Title</b></i><br />...
```

When I ran html2text it produced an output like this:

```
Some text.
 _ **Title**_
...
```

This isn't correct markdown; there shouldn't be a space after the underscore.

I was pretty unlucky to hit this so quickly.
Having only one type of emphasis or whitespace between the `.` and `<i>` tag would produce correct output.
So once I found a minimum reproducible example I [raised an issue](https://github.com/Alir3z4/html2text/issues/332).

## Understanding the issue

The html2text library was originally written by [Aaron Swartz](https://en.wikipedia.org/wiki/Aaron_Swartz), the inventor of [Markdown](https://daringfireball.net/projects/markdown/).
The bulk of it is implemented as a very large Python [html.parser](https://docs.python.org/3/library/html.parser.html) object.

The issue comes in how whitespace is handled in the following code:

```python
def no_preceding_space(self: HTML2Text) -> bool:
    return bool(
        self.preceding_data and re.match(r"[^\s]", self.preceding_data[-1])
    )

if tag in ["em", "i", "u"] and not self.ignore_emphasis:
    if start and no_preceding_space(self):
        emphasis = " " + self.emphasis_mark
    else:
        emphasis = self.emphasis_mark

    self.o(emphasis)
    if start:
        self.stressed = True

if tag in ["strong", "b"] and not self.ignore_emphasis:
    if start and no_preceding_space(self):
        strong = " " + self.strong_mark
    else:
        strong = self.strong_mark

    self.o(strong)
    if start:
        self.stressed = True
```

When there's not a preceding space one is added.
However *this* inserted space in the output should count as preceding, but it doesn't, and so another space is added.

Since this is the only usage of `preceding_data` a safe workaround is to append a space to `self.preceding_data`.
There are many other ways we could possibly preserve the state if altering this may break existing code.
I've tried submitting this as a [pull request](https://github.com/Alir3z4/html2text/pull/333).

Parsing HTML in a meaningful way is really hard!
The html2text library is very mature and does very well, but even it has weird edge cases.