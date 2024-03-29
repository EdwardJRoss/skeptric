---
categories:
- python
- programming
date: '2020-11-29T21:26:24+11:00'
image: /images/bs4_getattr.png
title: Type Checking Beautiful Soup
---

Static type checking in Python can quickly verify whether your code is open to certain bugs.
But it only works if it knows the types of external libraries.
I've already [introduced how to add type stubs](/python-type-stubs) for libraries without type annotations.
But what if we have a complex library like [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/) that uses a lot of recursion, magic methods and operated on unknown data?

With some small changes to your code you can make it typecheck with BeautifulSoup.
This helped me catch some places where I could have got an error when None was returned (the [Billion Dollar Mistake](https://www.infoq.com/presentations/Null-References-The-Billion-Dollar-Mistake-Tony-Hoare/)).
The naive way to do this requires peppering your code with assertions, but a much better way is to use CSS selectors.

Consider something that extracts a the text from a h1 tag like this:

```python
import bs4

soup = bs4.BeautifulSoup(html, "html5lib")
data = soup.h1.b.get_text()
```

In the [source](https://bazaar.launchpad.net/~leonardr/beautifulsoup/bs4/view/head:/bs4/element.py) this is resolved with `__getattr__`:


```python
    def __getattr__(self, tag):
        """Calling tag.subtag is the same as calling tag.find(name="subtag")"""
        #print("Getattr %s.%s" % (self.__class__, tag))
        if len(tag) > 3 and tag.endswith('Tag'):
            # BS3: soup.aTag -> "soup.find("a")
            tag_name = tag[:-3]
            warnings.warn(
                '.%(name)sTag is deprecated, use .find("%(name)s") instead. If you really were looking for a tag called %(name)sTag, use .find("%(name)sTag")' % dict(
                    name=tag_name
                )
            )
            return self.find(tag_name)
        # We special case contents to avoid recursion.
        elif not tag.startswith("__") and not tag == "contents":
            return self.find(tag)
        raise AttributeError(
            "'%s' object has no attribute '%s'" % (self.__class__, tag))
```

I could annotate this, but if I don't annotate *all* the other functions and attributes this could lead to incorrectly inferred attributes, and that's a lot of annotation.
Instead I could just use my code above to use `find` directly.

```python
data = soup.find("h1").find("b").get_text()
```

The BeautifulSoup code has some sort of type annotations in the docstring which makes it much easier to annotate.
It tells us the return type of `find` is `bs4.element.Tag | bs4.element.NavigableString`.
Looking at the implementation it just returns the first results of `find_all`, or `None` if there isn't one.
The docstring for `find_all` tells us that it's a `ResultSet` (a subclass of list) of `PageElements`.
It turns out that `Tag` and `NavigableString` are `PageElements` so this more or less lines up.
So putting this together we have a first implementation of a type stub at `bs4/element.pyi`

```python
from typing import Optional

class PageElement:
  pass

class Tag(PageElement):
    def find(
        self,
        name: Optional[str] = None,
        attrs={}, recursive=True, text=None, **kwargs
    ) -> Optional[PageElement]: ...
    
    def get_text(
        self, separator: str = "",
        strip: bool = False
    ) -> str: ...
```

Trying to typecheck now comes up with an error:

```
error: Item "None" of "Optional[PageElement]" has no attribute "find"
```

This makes a good point; what if the page *doesn't* have an `h1`?
I have to do something, perhaps in this case I know my pages will contain a `h1` and so this isn't a problem.
The same is true for the `b` inside the `h1`, but this is more common, and maybe we have a default action here.
I can either suppress the type error with an explicit assertion (for the `h1`) or explicitly handle the `None` case (for the `b`):

```python
h1_tag = soup.find("h1")
assert h1_tag is not None
h1b_tag = h1_tag.find("b")
if h1b_tag:
    data = h1b_tag.get_text()
else:
    data = None
```

But now I get a different error:

```python
error: "PageElement" has no attribute "find"
error: "PageElement" has no attribute "get_text"
```

Neither `PageElement` nor `NavigableString` have a `find` or `get_text` method.
But I can't actually see with this usage how I can get a `NavigableString` (maybe I can with a certain set of arguments to find).
So I could explicitly declare this with an ugly bunch more assertions:

```python
h1_tag = soup.find("h1")
assert h1_tag is not None
assert isinstance(h1_tag, bs4.Tag)
h1b_tag = h1_tag.find("b")
if h1b_tag:
    assert isinstance(h1b_tag, bs4.Tag)
    data = h1b_tag.get_text()
else:
    data = None
```

This is pretty horrible.
But there's a better way with [CSS selectors](https://developer.mozilla.org/en-US/docs/Learn/CSS/Building_blocks/Selectors).
They can represent all sorts of complex relationships with [combinators](https://developer.mozilla.org/en-US/docs/Learn/CSS/Building_blocks/Selectors/Combinators), in this case to find a bold element in a h1 the selector is `h1 b` (using the descendent combinator).
By constructing some path with CSS Selectors we only need to check whether the whole path is null, not every step of the way.
CSS Selectors are a similar power to [XPath](https://en.wikipedia.org/wiki/XPath), but you can test them in a browser console with [`document.querySelectorAll`](https://developer.mozilla.org/en-US/docs/Web/API/Document/querySelector)

BeautifulSoup has CSS selector methods; `select` for multiple results, and `select_one` for the first result (or None if there aren't any).
According to the types in the code they always return `Tag` so there's no need to worry about `NavigableString` or other types of `PageElement`.
We can add these to our type stub:

```python
class Tag(PageElement):
    def select_one(
        self, selector: str, namespaces: Optional[Dict[str, str]] = None, **kwargs: str
    ) -> Optional[Tag]: ...
    def select(
        self, selector: str, namespaces: Optional[Dict[str, str]] = None, **kwargs: str
    ) -> List[Tag]: ...
```

Then we have a much simpler method, that gets a good tradeoff between simplicity and capturing the cases where the tag is absent.

```python
h1b_tag = soup.select_one("h1 b")
if h1b_tag:
  data = h1b_tag.get_text()
else:
  data = None
```

That's enough to get a long way with type checking BeautifulSoup.
For more check out my [tips on using BeautifulSoup](/beautiful-soup-tips).