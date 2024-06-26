---
categories:
- nlp
- data
date: '2020-11-03T21:03:54+11:00'
image: /images/wordcloud.png
title: Building NLP Datasets from Scratch
---

There's a common misconception that the best way to build up an NLP dataset is to first define a rigorous annotation schema and then crowdsource the annotations.
The problem is that it's actually really hard to *guess* the right annotation schema up front, and this is often the hardest part on the modelling side (as opposed to the business side).
This is explained wonderfully by [spaCy's Matthew Honnibal at PyData 2018](https://www.youtube.com/watch?v=jpWqz85F_4Y).

So what is the best way to build a labelled NLP dataset to solve a problem?
I believe it's using anything you can to get an initial labels to get the labels quickly, then refine the labels, test the dataset and iterate.
Only once you've proved a solution do you invest in large annotation (and only if you can prove it has business value).

There are always ways to get rough labels.
Heuristics based on domain knowledge.
Rule based methods from the dependency parse (which are marvellously accurate with modern neural dependency parsers).
Taking an external labelled dataset and modelling the labels onto your dataset.
Using complementary data, such as behavioural data (think matrix factorisation) to map the data onto a space.

Once you have rough labels you can then refine them.
You can use data programming to merge data from multiple noise labellers to get proposed labels.
You can train models using the probabilistic labels to get an initial classifier.
You can then use active learning to quickly improve performance, and get to a model good enough to check viability.

I don't believe there are any good resources on how to do this whole process from end to end, but it's very valuable to be able to take unlabelled text and within a few days have a process for extracting meaningful information from it.