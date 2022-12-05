---
categories:
- nlp
- books
- annotation
date: '2022-07-31T14:11:49+10:00'
image: /images/hil_headline_sport_results.jpg
title: 'Human-in-the Loop: Finding Topics in Headlines'
---

The book [Human-in-the Loop Machine Learning](/book-review-human-loop-machine-learning) by Randall Munro has a code example of [annotating headlines](https://github.com/rmunro/headlines) for a data analyst.

First you choose a topic name and then can annotate examples.
For example I chose "sports results" intending to label headlines containing the result of a sport contest (and not other kinds, e.g. political contests).
It wasn't totally obvious how to annotate examples at first; I had to click in the box with the example headline.

At the start I tried labelling random examples, which was very tiresome since most headlines were about other things.
I found that due to "repetition priming" (as described in the book) I would accidentally label a headline as negative.
There is no way to undo an annotation through the interface, which was quite annoying.

It has the interesting option to filter on terms and so I could get more relevant terms like "points", "final", or "win".
After enough items are labelled I ticked the "Focus on model uncertainty (if available)" which increasingly gave interesting examples.
There was no indication of progress, or how close we were (I had to check the terminal to see if a model was being used).

Unfortunately at times the model performed quite badly due to the model update strategy.
While using random sampling (no keyword or focus on model uncertainty), 1/4 of the items are randomly assigned to evaluation until you get a total of 50 when it stops.
In 50 results there was only 1 positive result which led to a very unbalanced evaluation set.
While it continually retrains the model it only updates the model when the f-score on the evaluation set improves; which was quite problematic with this evaluation set!
A simpler approach would have been much better here.

When a model was getting new model predictions getting new items to annotate became noticably slower on my laptop.
This is because it only yields to other threads every 50 predictions, and a submission and fetching a new example each required a yield.
Fixing this would be a huge quality of life improvement.

```python
        if count%50 == 0:
            eel.sleep(0.01) # allow other processes through
            if verbose:
                print(count)
```

One other useful feature was to be able to mark "interesting examples", which immediately gave an alert indicating the item had been marked as interesting.
Finally you could "show examples" by year, which worked pretty well after 40 positive annotations, and 330 negative annotations (helpfully they are stored in separate CSVs in the format Article Date,Headline,URL,Sampling Strategy).

Overall this was a really interesting example of Human-in-the-Loop Machine Learning.
A small decision only to update when the model's f1 score increased, and a bad choice of evaluation set, made the experience very bad.
Also the inability to go back, and a slowdown during evaluation made it much harder to use.
If anything I think the sampling approach is too conservative; I'd look for more ways to extract likely matches faster and use uncertainty sampling to make sure we're not focussing only on common cases.