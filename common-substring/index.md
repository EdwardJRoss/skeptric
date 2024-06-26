---
categories:
- data
- nlp
- jobs
date: '2020-05-17T19:34:05+10:00'
image: /images/common_substring.png
title: Finding Common Substrings
---

I've found pairs of near duplicates texts in the [Adzuna Job Salary Predictions Kaggle Competition](https://www.kaggle.com/c/job-salary-prediction) using [Minhash](/minhash-lsh).
One thing that would be useful to know is what the common sections of the ads are.
Typically if they have a high 3-Jaccard similarity it's because they have some text in common.

The most asymptotically efficient to find the longest common substring would be to build a [suffix tree](https://en.wikipedia.org/wiki/Generalized_suffix_tree), but for experimentation the heuristics in Python's [DiffLib](https://docs.python.org/3/library/difflib.html) work well enough.

I define a function that gets all common strings above a certain length.
I look for all pairs difflib considers equal and print them out; this won't get all common substrings but works well enough on the job ads I tried them on.
A benefit of using difflib is that if we want to find the longest common string of *tokens* we can just pass in `a` and `b` as lists of tokens.

```python
def common_substrings(a, b, min_length=7):
    seqs = []
    seqmatcher = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    for tag, a0, a1, b0, b1 in seqmatcher.get_opcodes():
        if tag == 'equal' and a1 - a0 >= min_length:
            seqs.append(a[a0:a1])
    return seqs
```

Once we have a way to find common substrings of pairs it's straightforward to extend it to a list of substrings.
In particular we just look to see for each existing common substrings whether it has any common substrings with the next text.

```python
def all_common_substrings(args, min_length=7):
    seqs = None
    for arg in args:
        if seqs is None:
            seqs = [arg]
            continue
        new_seqs = []
        for seq in seqs:
            new_seqs += common_substrings(arg, seq, min_length)
        seqs = new_seqs
    return seqs
```

This could easily be extended to allow a few mismatches by collecting across difflib tags other than `equal` up to some length of tokens.

# Exact brute force approach

It's good to compare this with an exact solution to make sure the difflib heuristics are actually working.
I always find it's good to start with a simple slow obviously correct solution before trying to build a more complex efficient algorithm.

In particular we could start by producing all substrings of a string by iterating over each possible starting point and length:

```python
def all_substrings(s):
    for i in range(len(s)):
        for j in range(len(s)-i):
            yield s[i:i+j+1]
```

To know whether one string is a substring of another we can just check whether it matches a any position:

```python
def contains_substring(a, b):
    """Does a constrin substring b"""
    for i in range(len(a) - len(b) + 1):
        if a[i:i+len(b)] == b:
            return True
```

Then we could find the common substrings by just checking if any of the substrings of one are in the other:

```python
def naive_common_substrings(a, b):
    for substring in all_substrings(a):
        if contains_substring(b, substring):
            yield substring
```

This will output a lot of substrings because any substring of a common substring is also a common substring.
For example if "the" is in common, then so is "t", "th", "h", "he" and "e".
We can filter this down to the "proper substrings", those that aren't contained in a larger substring.

```python
def proper_substrings(a):
    proper = []
    for s in a:
        if any(contains_substring(p, s) for p in proper):
            continue
        supersequence = [contains_substring(s, p) for p in proper]
        if any(supersequence):
            val = s
            for idx, value in enumerate(supersequence):
                if value:
                    proper[idx] = val
                    val = None
            proper = [p for p in proper if p]
        else:
            proper.append(s)
    return proper
```

Note that this implementation is awfully slow; the operations in calcualting `proper_substrings(naive_common_substrings(a, b))` is quadratic in the length of `a` and roughly linear in the length of `b`.
But it's good for a sanity check on some simple strings, and using it I find the `difflib` captures most of the common substrings on the job ads I tried.