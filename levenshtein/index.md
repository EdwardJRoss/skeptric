---
categories:
- nlp
- python
- jobs
date: '2020-04-17T08:31:53+10:00'
image: /images/ads_process_project_engineer.png
title: Edit Distance
---

Edit distance, also known as [Levenshtein Distance](https://en.wikipedia.org/wiki/Levenshtein_distance) is a useful way of the similarity of two sequences. 
It counts what is the minimum number of substitutions, insertions and deletions you need to make to transform one sequence to another.
I had a look at using this for trying to compare duplicate ads with reasonable results, but it's a little slow to run on many ads.

I've previously looked at finding [ads with exactly the same text](/exact-duplicates.mmark) in the [Adzuna Job Salary Predictions Kaggle Competition](https://www.kaggle.com/c/job-salary-prediction), but there are a lot of ads that are slight variations.
 

The Python library [editdistance](https://github.com/aflc/editdistance) has a fast implementation and supports and iterable with hashable elements.
This means for comparing text we can pass in a whole string for a character-wise edit distance, or we can tokenise it into a list of words for a word-wise edit distance.

Job ads differ in length dramatically, so I wanted to know how different they were relative to the largest one making a relative_editdistance.
Identical texts will have a relative edit distance of 0 and texts that are completely different will have a relative edit distance of 1.


```python
import editdistance

def relative_editdistance(a, b):
    return editdistance.eval(a, b) / max(len(a), len(b))
```

Then I could compare the ads (here characterwise) with a double loop; it took 50s on my laptop for 100 ads.

```python
distance = {}
ads_sample = ads[:100]
for i, ad1 in enumerate(ads_sample):
    for j, ad2 in enumerate(ads_sample):
        if i < j:
            distance[(i, j)] = relative_editdistance(ad1, ad2)
```

Then looking at the most similar ads I could find near duplicates easily:

![Similar ads for project and process engineer](/images/ads_process_project_engineer.png)

However to run this on the full 400k ads will take 36 years, because this scales quadratically.
For identical ads I could sort them (or their hash), but this doesn't work here because they could be different anywhere in the string (and often *are* different at the start and the end).

In then next part of the series I'll look at using MinHash to solve the duplicate problem at scale.