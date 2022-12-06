---
categories:
- books
- annotation
date: '2022-08-10T08:00:00+10:00'
image: /images/food_safety_annotation.jpg
title: 'Human-in-the Loop: Finding Hazards in Food'
---

The book [Human-in-the Loop Machine Learning](/book-review-human-loop-machine-learning) by Randall Munro has a code example of [finding hazards in food safety reports](https://github.com/rmunro/food_safety).
Here's the description of the problem:


> Food Safety professionals want to collect data from incident reports about where pathogens or foreign objects have been detected in food.
>
> - “I want to maintain a complete record of all recorded food safety incidents in the EU”
> - “I want to track when different food safety incidents might have come from the same source”
> - “I want to send warnings to specific countries when there are likely to be food safety incidents that have not yet been detected or reported”

The interface has fields for "Hazard", "Food", "Origin", and "Destination" along with a short extract of text from a food report.
When you start typing in any of the fields a list of possible completions shows below the text, which can be navigated with arrow keys and selected with enter.
You can navigate between the fields using Tab and Shift-Tab, and submit it by pressing Save and then enter.
Because Save is a separate process I never felt the need to "undo" an annotation, and the hotkeys were quite good.
You could also leave fields blank when there wasn't the information present; sometimes the origin or destination was missing, or sometimes there wasn't a hazard (e.g. "suspected fraud").
This seems like a good interface for high precision annotation - it helps you with the task, but isn't suggestive and you can always override it.
There were many cases where I misread a field (e.g. France instead of Finland), and only noticed because of the autocomplete - so it likely increased the precision of my annotations.

There were some issues in the user interface that made it a bit harder to use.
When I tried to select an autocomplete with an apostrophe it only populated the text before the apostrophe in the field; however because text could be manually entered I could work around this manually (always good to have manual overrides).
The autocomplete for the previous field stayed populated when I selected the next field, which made it harder to read and sometimes scrolled past the extract from the food safety report (so I'd have to manually scroll back up again).
When the first model trained the suggestions were actually much worse than the ngram matching model; there probably should have been a higher threshold to switch to the model.
Also the initial model download blocked all the threads including annotation (with no feedback) which was frustrating.
Finally retraining failed after some time so the model stopped getting better; in the console I got the error:

```
ValueError: Expected input batch_size (48) to match target batch_size (47).
```


Overall this was an interesting example of a Human-in-the-Loop model and really demonstrated what you could achieve with this kind of interface.
At the bottom it showed other similar annotations (with the same hazard, origin, food, or destination) which could really be useful for a food safety expert to notice linked outbreaks.
The ambiguity in the task quickly became clear (do I annotate the common name, or the scientific name or elaboration?), which would need refinement with many annotators.
My main lesson is the annotator experience of using these tools is crucial; if you want people to spend a long time in these tools you need to get these tools into people's flow and make them as enjoyable to use as possible, even small frictions cause pain.
More importantly there should always be fallbacks; being able to override the autocomplete allowed working past bugs that would otherwise completely block the process.
