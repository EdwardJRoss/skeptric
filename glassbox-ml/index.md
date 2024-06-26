---
categories:
- data
date: '2020-12-08T21:36:09+11:00'
image: /images/ml-method-timeline.png
title: Glassbox Machine Learning
---

Can we have an *interpretable* model that has as good performance as blackbox models like gradient boosted trees and neural networks?
In a [2020 Empirical Methods for Natural Language Processing Keynote](https://slideslive.com/38942829/emnlp-live-keynote2), Rich Caruana says yes.

He calls interpretable models *glassbox machine learning*, in contrast to blackbox machine learning.
It is models in which a person can explicitly see how they work, and follow the steps from inputs to outputs.
This interpretability is subtly different from explainable (explainable to who?), or plausible (to who?) and quite orthogonal to causal.

Examples of glassbox models are decision trees, (general) linear models, and k-nearest neighbours.
There's a good chapter on interpretable models in [Christoph Molnar's Interpretable Machine Learning book](https://christophm.github.io/interpretable-ml-book/simple.html).
Other notable examples are [Cynthia Rudin's](/interpretable-models-rudin) [Falling Rule Lists](https://arxiv.org/abs/1411.5899) (a sort of very simple decision tree) and [risk-slim risk scores](https://github.com/ustunb/risk-slim).
However these methods are generally considered to be less predictive than gradient boosted trees and neural networks.

Rich Caruana talks about *Explainable Boosting Models* as a glassbox model that works almost as well as the best blackbox models.
It originates from a 2012 KDD paper, [Intelligible Models for Classification and Regression](http://www.cs.cornell.edu/~yinlou/papers/lou-kdd12.pdf), where they fit a particular type of Generalised Additive Model.
For the shape function they use bagged trees on each feature and use a gradient boosting method to fit the trees, with good results.
In a 2013 KDD paper, [Accurate Intelligible Models with Pairwise Interactions](http://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf), they introduced pairwise interactions between features which greatly increased the accuracy of the models, while still remaining interpretable.
Like all GAMs you end up with a sum of graphs (in this case with 1 or 2 independent variables), each of which can be inspected to understand how the model makes predictions.
There's a 2015 paper, [Intelligible Models for HealthCare: Predicting PneumoniaRisk and Hospital 30-day Readmission](https://www.microsoft.com/en-us/research/wp-content/uploads/2017/06/KDD2015FinalDraftIntelligibleModels4HealthCare_igt143e-caruanaA.pdf) which shows how this can be used to solve real problems.
Interestingly the glassbox methods help identify data issues which may escape an exploratory data analysis.
    
A challenge of this method is you need good features; it doesn't work well if you've just got a bunch of text, images, or unidentified columns because you can't interpret (or view all of) the graphs.

> If you don't know what the feature means then you don't know what it implies

As I discussed in [rules and models](/rules-and-models) the most promising way forward seems to be to build features on top of the raw data (using flexible blackbox methods like neural networks), and then build an explainable model on top of that - but it's a lot of work.

Another challenge is collinearity.
If two features are the same how do you share the weight of their contribution to the sum?
Explainable Boosting Models share the load equally - which is probably the least bad thing to do.

This all sounds really interesting and I'm keen to try it in the [InterpretML](https://github.com/interpretml/interpret) tool on real datasets to see how it performs against good baselines (e.g. Kaggle competitions).