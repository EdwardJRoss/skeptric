---
categories:
- nlp
- jobs
date: '2020-06-06T21:14:45+10:00'
image: /images/singular_suffix.png
title: Making Words Singular
---

Trying to normalise text in [job titles](/job-title-words) I need a way to convert plural words into their singular form.
For example a job for "nurses" is about a "nurse", a job for "salespeople" is about a "salesperson", a job for "workmen" is about a "workman" and a job about "midwives" is about a "midwife".
I developed an algorithm that works well enough for converting plural words to singular without changing singular words in the text like "*sous* chef", "business" or "gas".

Normally I would use a library for something like this but I couldn't find anything that would work.
The [inflect](https://github.com/jazzband/inflect) library has a `singularize_noun` function that works on plural words, but it has no way of detecting whether a noun is plural and [changes singular words](https://github.com/jazzband/inflect/issues/76).
Similarly [textblob](https://textblob.readthedocs.io/en/dev/) has a `singularize` function, but it is largely [based on inflect](https://github.com/sloria/TextBlob/blob/dev/textblob/en/inflect.py) and has the [same issues](https://github.com/sloria/TextBlob/issues/281).
Some examples are "gas" becomes "ga", "bus" becomes "bu" and "analysis" becomes "analysi".

One way to deal with this is to use a part of speech tagger, like SpaCy's, to identify the plural nouns.
If the part of speech is NNS (plural noun) or NNPS (plural proper noun) then we run `singularize_noun`.
However bringing in a POS tagger seems like a big thing for a simple task, and it's going to be hard to fix where the part of speech is misclassified.
I wanted to see if I could build some simple rules to do it myself.

Between the infect library and this [article on plurals](http://users.monash.edu/~damian/papers/HTML/Plurals.html) there is a lot of material on building a robust algorithm to make words plural or singular.
But I needed something simple for my use case, so I started with the simplest rule, drop the trailing "s".
Then I looked through the most common words in my dataset to find exceptions and infer rules; like bus, plus, fabulous and sous I wonder if ending in "us" is a general exception.
Then I would look through the most frequent words ending in "us" and find the rule applies except for in "menus".

Through this I built up a list of rules and test cases:

| Original Form | Singular Form | Original Suffix | Singular Suffix | Original Plural |
|---------------|---------------|-----------------|-----------------|-----------------|
| accounts      | account       | s               | (None)          | Yes             |
| executives    | executive     | es              | e               | Yes             |
| nannies       | nanny         | ies             | y               | Yes             |
| midwives      | midwife       | wives           | wife            | Yes             |
| sous          | sous          | us              | us              | No              |
| menus         | menu          | us              | u               | Yes             |
| business      | business      | ss              | ss              | No              |
| analysis      | analysis      | is              | is              | No              |
| workmen       | workman       | men             | man             | Yes             |
| salespeople   | salesperson   | people          | person          | Yes             |
| sales         | sales         | sales           | sales           | No              |
| children's    | children's    | 's              | 's              | No              |
| gas           | gas           | gas             | gas             | No              |
| geophysics    | geophysics    | physics         | physics         | No              |
| asbestos      | asbestos      | asbestos        | asbestos        | No              |

Note these rules are applied with the most specific rule first; for example nannies ends in "ies" and "es"; but "ies" is more specific so we get nanny and not nanne.
These are nowhere near comprehensive but are a good start for normalising role titles.

I then build these rules into a function:

```python
SINGULAR_UNINFLECTED = ['gas', 'asbestos', 'womens', 'childrens', 'sales', 'physics']

SINGULAR_SUFFIX = [
    ('people', 'person'),
    ('men', 'man'),
    ('wives', 'wife'),
    ('menus', 'menu'),
    ('us', 'us'),
    ('ss', 'ss'),
    ('is', 'is'),
    ("'s", "'s"),
    ('ies', 'y'),
    ('ies', 'y'),
    ('es', 'e'),
    ('s', '')
]
def singularize_word(word):
    for ending in SINGULAR_UNINFLECTED:
        if word.lower().endswith(ending):
            return word
    for suffix, singular_suffix in SINGULAR_SUFFIX:
        if word.endswith(suffix):
            return word[:-len(suffix)] + singular_suffix
    return word
```

There are numerous places this will fail; wolves, wives, oxen, crises and many others.
These rules could be further expanded using the rules from inflect, but you would have to be careful not to break singular words.
For example it has a rule "lves" -> "ves" which is fine for wolves, calves and shelves, but will break evolves and involves.

There are also some proper nouns in my data, like Leeds and Wales, that neither algorithm can handle.
This may be where a part of speech approach may be more powerful.

However this simple set of rules is useful enough for normalising role titles in the Adzuna job ads.