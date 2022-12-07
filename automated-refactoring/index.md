---
categories:
- python
- legacy code
date: '2021-06-01T21:28:40+10:00'
image: /images/extract_method.png
title: Automated Refactoring in Python
---

I am a very recent convert on automatic refactoring tools.
I thought it was something for languages like Java that have a lot of boilerplate, and overkill for something like Python.
I still liked the concept of refactoring, but I just moved the code around with Vim keymotions or sed.

But then I came up against a giant Data Science codebase that was a wall of instructions like this:

```python
import pandas as pd
import datetime

df = pd.read_csv('data.csv') 

# Get the current age
now = datetime.datetime.now()
df['age'] = now - df['date_of_birth']
df['age'] = df['age'].clip(20, 100)

# Convert the height from inches to cm
df['height_cm'] = (df['height'] * 2.54)
df['height_cm'] = df['height_cm'].round().astype('Int64')
# Impute missing height values with the mean
df['height_cm'].fillna(df['height_cm'].mean())

# Recenter the data
...
```

The problem with this is it's very hard to follow what's going on, and it's very hard to test.
The solution is [comment to function](/comment-to-function) where we replace all the sections starting with a comment with a function, more like:

```python
df = pd.read_csv('data.csv') 

df['age'] = get_age(df['birth_year'], datetime.now())

df['height_cm'] = inches_to_cm(df['height'])
df['height_cm'] = impute_with_average(df['height_cm'])
...
```

This makes the high level logic easier to follow, and each function can have independent unit tests.

I did the first of these refactors by hand, manually copying the code, identifying the parameters, copying them up to the function signature and then renaming them.
I'd often get it wrong the first time and miss a parameter, or get them in the wrong order.

Then I tried an automated "Extract Method" refactoring in VSCode, and it just worked.
It handled working out the parameters and creating the function.
This took a lot of cognitive load off of me, made be work faster and be more aggressive with the extractions.
It's a small thing, but in this kind of circumstance I really see the value of these automated extractions.

VSCode doesn't have an automated way to reorder the signature built in, and the default order was often bad, and so I would fix the signature in the call, and then copy the new signature down to the definition.
Often in the definition I'd want to name the parameters differently, and so I could use rename symbol to change them all.

There are a number of tools for these kinds of things in Python, here are a few:

* [PyCharm](https://www.jetbrains.com/help/pycharm/refactoring-source-code.html) has a very large set of refactorings built in
* [VSCode](https://code.visualstudio.com/docs/python/editing#_refactoring) has a few basic refactorings built in
* [Jedi](https://jedi.readthedocs.io/en/latest/index.html) package has a few refactoring methods
* [Rope](https://github.com/python-rope/rope/blob/master/docs/overview.rst#refactorings) has many more refactorings

I'm not sure about how they all compare, but I'm going to experiment with them more.