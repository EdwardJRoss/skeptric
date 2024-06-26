---
categories:
- python
- testing
date: '2021-07-02T14:41:51+10:00'
image: /images/regex_hypothesis.png
title: Property Based Testing with Regular Expressions
---

[Property based testing](/property-based-testing) is a really useful technique where you state a property about your code and then verify it with random data.
The difficulty is generating good random data that will thoroughly exercise your code.
For text data you can do this by generating text, in particular with regular expressions.

While you could easily do this with hypothesis [from_regex strategy](https://hypothesis.readthedocs.io/en/latest/data.html#hypothesis.strategies.from_regex) you could roll your own with Hypothesis and the library [rstr](https://github.com/leapfrogonline/rstr) as follows:

```python
from hypothesis.strategies import composite, randoms
from rstr import Rstr

@composite
def text_from_regex(draw, regexp):
    random = draw(randoms())
    return Rstr(random).xeger(regexp)
```

I came up with this when I had a tricky regex to try to parse salary expressions from strings.
I found a particular case where the regex returned something that wasn't a number and raised an error when it tried to get a float.

The first thing I tried was to generate random text to reproduce the error:

```python
from hypothesis import given, strategies as st

@given(text())
def test_extract_salary_types(text):
    salary_range = extract_salary(text)
```

But it couldn't reproduce the error, even after tens of thousands of test cases because it wasn't generating the right kind of text.
I also tried [Crosshair](https://github.com/pschanely/CrossHair) to find a failing case but it also failed to (maybe because it's hard to parse the regex).

Then I realised the regex I was using to parse would be a good hint on the kinds of things to try.
I found two libraries that can generate text from regex [rstr](https://github.com/leapfrogonline/rstr) and [exrex](https://github.com/asciimoo/exrex), and the former seemed easier to plug into Hypothesis (but exrex looks interesting because it can generate cases in increasing complexity, which would be good for shrinking).

As part of my expression I had a blacklist of things I tried to exclude from the regular expression (BLACKLIST_RE) and so tried this:

```python
@given(text_from_regex(regexp=rf"([\$£€\d\w\s-]|{BLACKLIST_RE})*"))
def test_extract_salary_types(text):
    salary_range = extract_salary(text)
```

This immediately came up with a counterexample `0a0m`.

This seems to be an interesting way to generate targeted examples to test your code against.
You could take this even further by generating text from a grammar using the [Hypothesis Lark extension](https://hypothesis.readthedocs.io/en/latest/extras.html?highlight=lark#hypothesis-lark).
In another direction you could use a language model trained on examples of your data, such as an n-gram language model, Hidden Markov Models or deep language models like BERT, ULMFiT or ELMO (and this is close to the ideas in [checklist](/nlp-checklist) and [textattack](https://github.com/QData/TextAttack)).
