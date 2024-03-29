---
categories:
- python
- html
- testing
date: '2022-06-17T22:34:58+10:00'
image: /images/html_comment_spec.png
title: Regular Expression for HTML Comments
---

I started trying to write a grammar for generating HTML and quickly got stumped by how to represent HTML comments.
From the [whatwg specification](https://html.spec.whatwg.org/multipage/syntax.html#syntax-comments):

> Comments must have the following format:
>
> 1. The string `"<!--"`.
> 2. Optionally, text, with the additional restriction that the text must not start with the string `">"`, nor start with the string `"->"`, nor contain the strings `"<!--"`, `"-->"`, or `"--!>"`, nor end with the string `"<!-"`.
> 3. The string `"-->"`.
>
> The text is allowed to end with the string `"<!"`, as in `<!--My favorite operators are > and <!-->`.

A starting point for this is the regular expression `<!--.*?-->`, but this doesn't cover all of the exclusions.
For parsing it's generally fine to be a bit more liberal, but I'm trying to generate HTML and so I want to be stricter.
The correct regular expression is `<!--(?!-?>)(?!.*--!>)(?!.*<!--(?!>)).*?(?<!<!-)-->`; the rest of this post aims to explain how you might get this.


Writing these kinds of regular expressions can be hard; fixing one case can break the others.
It's good to work with a set of test cases to check.
Here's a set I continually tested against.

```python
correct = [
    "<!---->",
    "<!----->",
    "<!--<-->",
    "<!--<!-->",
]

wrong = [
    "<!-->-->",
    "<!--->-->",
    "<!-- <!-- -->",
    "<!-- --!> -->",
    "<!--<!--->",
    "<!---->-->", # Matches <!---->
    "<!--->",
]
```

The first modification we can make is to remove those starting with `>` or `->` using a negative lookahead `(?!-?>)` at the start of the string.
It's a bit confusing because the special regex characters are similar to the HTML comment characters; the `(?!` signals the string is a negative lookahead, and the `-?>` says it should reject an optional `-` followed by `>`.
To ensure it doesn't end with `<!-` we can use a negative lookbehind at the end of the string `(?<!<!-)` (similarly the `(?<!` denote the negative lookbehind so it rejects strings preceeded by `<!-`).
We don't need to worry about `"-->"` inside the string since we will stop when we hit these tokens.
The only thing left is the forbidden containments: `"<!--"`, and `"--!>"`,

One way to handle containments is with the negative lookahead with a wildcard prefix, for example `(?!.*<!--)`.
However this means it will search through the *whole document* for this string; so it won't match on a pair of comments `<!----> <!---->`.
Instead we need to be more selective than `.*?`.
Any characters other than `-` and `<` are alright so we can safely use `[^<-]*`.
And a `-` is fine as long as it's not followed by `-!>`, which gives `-(?!-!>)`.
And `<` is fine as long as it's not followed by `!--`, which gives `<(?!!--)`.
So this gives us `([^<-]|-(?!-!>)|-(?!-!>)`

When I run everything so far I lose one of our correct expressions `<!--<!-->`.
This is because in our last rule when we parse the second `<` we see `!--`, not noticing that the second hyphen is part of the comment close.
So we can nest this condition as a negative lookahead inside the negative lookahead `(?!>)`.

Let's put this all together in a Python regular expression using the [verbose flag](https://docs.python.org/3/howto/regex.html#using-re-verbose):

```python
import re

html_comment = re.compile(r"""
<!--              # Open comment
(?!-?>)           # Can't start with - or ->

(?:
  [^<-]         |  # Any character other < or -  
  -(?!-!>)      |  # - not followed by -!>
  <(?!!--(?!>)) |  # < not followed by !-- (except at end)
)*?

(?<!<!-)          # Can't end with <!-
-->               # End comment
""", re.VERBOSE)
```

This is pretty complicated; how do we know we didn't miss any examples?
We can try using [property based testing](/property-based-testing) with Hypothesis.

Let's rewrite whether a string is a comment based on the rules above.
Since the rules are mostly about excluding comments if it's not a comment we will return an integer to identify the rule it failed on to make debugging easier.
If it is a comment we will return None.

```python
from typing import Optional

def not_comment(s: str) -> Optional[int]:
    comment_start = '<!--'
    comment_end = '-->'
    
    if not s.startswith(comment_start):
        return 1
    
    if not s[len(comment_start):].endswith(comment_end):
        return 2
    inner = s[len(comment_start):-len(comment_end)]
    
    if inner.startswith(">"):
        return 3
    if inner.startswith("->"):
        return 4
    if "<!--" in inner:
        return 5
    if  "-->" in inner:
        return 6
    if "--!>" in inner:
        return 7
    if inner.endswith("<!-"):
        return 8
    
    return None
```

We can then test whether these two give the same results.
While we could test on any random string the search space for counterexamples is huge.
Instead we'll test for things close to a regex, starting with `<!-` and ending with `->`.
Moreover we can test all the cases with just the characters `<!->` since the rules only apply to them.
So we'll generate examples from a regex `<!-[<!->]*->`:

```python
from hypothesis import given
import hypothesis.strategies as st

@given(st.from_regex(r"<!-[<!->]*->", fullmatch=True))
def test_comment(comment):
    result = html_comment.match(comment) is None
    expected = not_comment(comment)
    assert result == bool(expected)
```

Runinng this with pytest I don't get any failures (and I do get failures when I delete rules).
Another test is that nothing after a comment should change the match (the wildcard negative lookaheads violated this).
We can test this in hypothesis too:

```python
from hypothesis import assume

@given(st.from_regex(r"<!--[<!->]*-->", fullmatch=True),
       st.text("<!->"))
def test_comment(comment, extra):
    assume(not not_comment(comment))

    result = html_comment.match(comment + extra)
    result = result.group() if result else None
    
    expected = html_comment.match(comment)
    expected = expected.group() if expected else None
    
    assert result == expected
```

This also succeeds in pytest, but using the negative lookeaheads gives useful failures.
So with all this I'm pretty confident the regular expression `<!--(?!-?>)(?:[^<-]|<(?!!--(?!>))|-(?!-!>))*?(?<!<!-)-->` matches HTML comments as per the specification.