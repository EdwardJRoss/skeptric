---
categories:
- annotation
date: '2022-06-30T08:00:00+10:00'
image: /images/disaster-annotation.png
title: Human in the Loop for Disaster Annotation
---

I've been reading Robert Monarch's [Human-in-the-Loop Machine Learning](https://www.manning.com/books/human-in-the-loop-machine-learning) and the second chapter has a great practical example of human-in-the-loop machine learning; identifying whether a news headline is about a disaster.
A lot of the Data Science books, courses, and competitions take for granted that you've got a well defined machine learning metric to optimise, and labelled dataset.
In practice you often have a practical objective, and have to define the metric and collect the input features and labels.
Sometimes you can use existing data, but other times you have to label a new dataset.
This book is great about talking about the process of labelling a dataset, which is a very human endeavour, and the example helps understand that.
I really recommend doing something like this to get a feel for the annotation process.

The author has made the [code available](https://github.com/rmunro/pytorch_active_learning); when I first tried to `pip install -r requirements.txt` in a fresh virtual environment I was told it couldn't find the dependencies, so I dropped the `==` restrictions from `requirements.txt` and it seemed to work fine.
Running `python active_learning_basics.py` dropped me into a command line interface where I could annotate examples.
There are a few commands which are submitted by typing and pressing `enter`.
No input labels the shown example as `not-disaster-related`; a `1` labels it as `disaster-related`, a `2` goes back to the previous example, `s` saves the annotations and `d` shows the detailed instructions.
Aside from save I needed all of these; most examples are `not-disaster-related` so saving a keystroke was a blessing, after a while my attention would wander and I'd need to go back to a previous example, and when I got to a tricky example I'd reread the detailed instructions.

The labelling process starts with random sampling, and then once enough data is trained it switches to a blend of uncertainty sampling (80%), diversity sampling (10%), and random sampling (10%).
At the start it's a real slog; there are very few disaster related examples (I labelled 17 in the first 300) and it's really searching for anything related to disasters.
After the first training it quickly gets more interesting; there's a lot more relevant examples (and some strange examples from the diversity sampling; many in Indonesian).
It's easier to pay attention and start to see themes.
After the second iteration it gets more interesting again, and I spend a lot more time looking back at the instructions.
When I first started annotating any mention of fire seemed like a disaster; but as I read more examples I started thinking about whether a house fire really is a disaster.
By the third iteration I got very interesting edge cases which really made me think.
At this stage I really needed to go back and really understand the outcomes we're trying to drive (for example are fears of a disaster "disaster-related", how big does the impact need to be for it to be disaster-related).
As a result of this reflection my annotations drift over time as my understanding of the task changes.

This kind of labelling is a human endeavour, and it has to work well with the human.
Annotating the same label over and over is error prone, as I tend to lose focus and fall into bias.
Uncertainty labelling gives a good mixture of positives and negatives, and really hard examples, which raises the conversation of what are we trying to determine.
It's a rewarding process seeing the more interesting examples come up.
A sprinkling of random and diversity sampled texts doesn't detract from the experience, and helps the model explore a wider space (which can lead to new conversations).

Each time a model is trained it's saved to a file with the date time, F1 score, AUROC, and number of labelled examples.
This gives a clear indication of how much training is improving, and is vindicating that even though I think I'm off the strict guidelines I'm still teaching the model useful things.
I ended up with files like this:

```
20220629_055234_0.0_0.528_400.params
20220629_060224_0.057_0.538_500.params
20220629_070931_0.109_0.562_600.params
20220629_085942_0.134_0.627_700.params
```

The interface matters a lot; it doesn't need to be complicated, but it needs to be clear.
I found the new text coming at the bottom of the screen to detract from my focus.
I also wish there was a way to abstain or flag examples; some were not in my language, and some were too ambiguous from the headline alone.
I was worried that starting with a randomly initialised model would take too much data to learn, but for this simple problem I got clear feedback at 500 parameters, and improvements every extra 100 parameters.
However I wonder how to tell when the task is too hard for the model to learn and needs to be broken down into smaller pieces (or if that's ever a problem for modern language models).