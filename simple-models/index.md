---
categories:
- data
date: '2020-05-16T19:31:03+10:00'
image: /images/constant_model.png
title: Simple Models
---

My first instinct when dealing with a new problem is to try to find a complex technique to solve it.
However I've almost always found it more useful to start with a simple model before trying something more complex.
You gain a lot from trying simple models and the cost is low.
Even if they're not enough to solve the problem (which they can be) they will often give a lot of information about the problem which will set you up for later techniques.

It's always worthwhile starting with the absolute simplest model possible to at least get a baseline for the problem.
I've seen people impressed by a complex system that gives 90% accuracy where a [constant model](/constant-models) will get in the 80% range; and the question should be does the extra 10 percentage points leads to significantly better decisions.
If you're looking at a [clustering problem](/clustering-segmentation) start off with everything in the same cluster, or with everything in a different cluster.
If you can clearly answer why the simplest model isn't good enough it's a big step towards evaluating more complex models (which is often harder than building them).

I've tried starting with a big black box technique like random forests before, but have found that I then spend a significant amount of time trying to get all the data and engineer the features to be in a columnar format; for example sequential data is hard to encode.
Then when the model didn't fit particularly well I was left stuck trying to work out where to go next.
If I were to try it again I would start by using domain knowledge to get the most likely few features and build a simpler model based on that, and incrementally build an understanding of the problem, before investing in gathering a large quantity of data.
I think there's still a place for these kinds of models and techniques like feature importance for discovering new angles to approach a problem, but it's worth ruling out simpler approaches first.

In reality an embarrassingly large number of business problems can be solved adequately with the right average or a linear model.
They're cheap to implement, explain and maintain and are robust in their domain.
In core business areas with high leverage, where a small improvement adds lot of business value, then it's generally worth improving these with more complex techniques.
But these will come with a much higher ongoing cost to implement, run, validate and maintain.

Keep in mind that "simple" is relative to your experience and the problem domain.
For an image classification problem the simplest reasonable model after a constant model is likely some kind of neural network.
But even [Andrej Karpathy recommends](http://karpathy.github.io/2019/04/25/recipe/) spending a lot of time looking at your data and getting dumb baselines with tiny models before putting in a ResNet.

Indeed I think a lot of the value of starting with simple models is the focus is more on the data and problem at hand, rather than on complex modelling or engineering issues, and so you get a much clearer understanding of what is likely to work and what the issues are.
In every setting you're likely to get bad data and this often stands out when you explore the data thoroughly.