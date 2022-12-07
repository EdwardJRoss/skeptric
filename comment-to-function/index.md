---
categories:
- programming
- legacy code
date: '2020-09-15T22:51:32+10:00'
image: /images/comment_to_function.png
title: Comment to Function
---

A lot of analytics code I've read is a very long procedural chain.
These can be hard to follow because the only way to really know what's going on in any point is to insert a probe to inspect the inputs and outputs at that stage.
Breaking these into functions is a really useful way of making the code easier to understand, change and find bugs in.

In Martin Fowler's [Refactoring](https://www.thoughtworks.com/books/refactoring2) he mentions that whenever there's a block of code that has (or requires) a comment to describe what it does, that's a good opportunity to package that code into a function.
I've found that a very good rule of thumb to follow; if the functions are well named it makes the code much clearer.
Here's a typical sort of simple example:

```python
# Read in the dataframe
df = pd.read_csv('data.csv', keep_default_na=False, low_memory=False)
# Convert temperatures to C
df['celcius'] = (df['temp']  - 32) * 5/9
# Lookup the temperature ranges
df = df.merge(temperature_range, how='left', left_on=['celcius'], right_index=True)
```

In a real example these blocks could be much longer and more obscure.
However if we package these up as functions the flow becomes a bit clearer:

```python
df = read_data('data.csv')
df['celcius'] = fahrenheit_to_celcius(df['temp'])
# Enrich modifies df
enrich_temperature_range(df, temperature_range)
```

By talking about what we're doing instead of how we're doing it the logic and dataflow becomes much clearer.
It also means these individual functions can be separately tested to ensure they're functioning correctly.
I've found lots of bugs and unreachable code blocks from this process of creating functions and breaking the code into small pieces.