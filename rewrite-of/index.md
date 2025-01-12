---
categories:
- nlp
date: '2020-06-05T22:59:21+10:00'
image: /images/a_of_b.png
title: Rewriting A of B
---

When examining [words in job titles](/job-title-words) I noticed that if was common to see titles written as "head of ..." or "director of ...".
This is unusual because most role titles go from specific to general (e.g. finance director) to you look backwards from the role word.
In the "A of B" format the role goes from specific to general and so you have to reverse the search order.
One solution is to rewrite "director of finance" to "finance director".

Here's my first cut of an algorithm:

```python
import re
def rewrite_of(term):
    word = r'[\w\d&]+' 
    next_word = fr'(?:\s+(?!for\s|or\s){word})'
    following = fr'{word}{next_word}' + '{0,3}'
    regex = fr'({word})\s+of(?:\s+the)?\s+({following})'
    return re.sub(regex, r'\2 \1', term, flags=re.IGNORECASE)
```

Note the negative lookahead *on*for and *word* to stop getting too much, and similarly for word excluding punctuation.

I'm not sure the regex approach is the best long term, but it's a useful starting point.
Iterating on this maybe me build a list of test cases, since a small change in the regular expression can make a big change in the output.

The easiest way to manage this is to build a suite of tests that check the result is as expected.