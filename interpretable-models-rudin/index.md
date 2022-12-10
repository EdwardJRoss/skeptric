---
categories:
- data
date: '2020-08-29T08:00:00+10:00'
image: /images/almost_matching_exactly.jpg
title: Interpretable models with Cynthia Rudin
---

A while ago I came across [Cynthia Rudin](https://users.cs.duke.edu/~cynthia/home.html) through their work on the [FICO Explainable Machine Learning Challenge](https://community.fico.com/s/explainable-machine-learning-challenge).
Her team got an [honourable mention](https://www.fico.com/en/newsroom/fico-announces-winners-of-inaugural-xml-challenge?utm_source=FICO-Community&utm_medium=xml-challenge-page) and she wrote an opinion with Joanna Radin [on explainable models](https://hdsr.mitpress.mit.edu/pub/f9kuryi8/release/5).
I think the article was hyperbolic on claiming interpretable models always work as well as black box models.
On the other hand I only came across her because of this article, so taking an extreme viewpoint in the media is a good way to get attention.

But getting past the headlines there is an interesting perspective.
Her tutorial on [The Secrets of Machine Learning](https://arxiv.org/pdf/1906.01998.pdf) takes a more nuanced point of view:

> most machine learning methods tend to perform similarly, if tuned properly, when the covariates have an inherent meaning as predictor variables (e.g., age, gender, blood pressure) rather than raw measurement values (e.g., raw pixel values from images, raw time points from sound files)

This is quite a reasonable claim; one of the benefits of black box models like boosted and bagged trees, and neural networks is that they do very well with complex raw features, and are currently nearly always a key part in winning machine learning competitions on Kaggle.
However when you have reasonable features you may be able to construct a simple model that can do quite well.
They go on to say:

> Interestingly, adding more data, adding domain knowledge, or improving the quality of data, can often be much more valuable than switching algorithms or changing tuning procedures

This is a great point; the amount of data (especially unusual examples) and the quality data fundamentally limit how good the model can be (and even how well you can evaluate model performance).
With simpler models it's easier to integrate domain knowledge to make a simpler model.
This has a larger up front cost than throwing a gradient boosting machine at the problem; but it has some potential benefits.

It's also important to consider the impact of performance on [decisions](/analysis-decision).
There is often diminishing returns and a small increase in model performance leads to a negligible increase in expected return.
This means that techniques common in machine learning competitions like ensembling and stacking models, which have a substantial operational and maintenance overhead, are often not worthwhile in practice.
That's not to mention the increased need for data quality to attain higher levels of performance.
The optimal tradeoff point depends on the amount of leverage you have; for Google's Ad products a small increase in click through rate can mean billions of dollars of revenue and it may be worth it (although these models are constrained by low latency).
This is an exceptional case.

The article concludes the section with a reasonable recommendation:

> Thus, the recommendation is simple: if you have classification or regression data with inherently meaningful (non-raw) covariates, then try several different algorithms. If several of them all perform similarly after parameter tuning, use the simplest or most meaningful model. Analyze the model, and try to embed domain knowledge into the next iteration of the model. On the other hand, if there are large performance differences between algorithms, or if your data are raw (e.g., pixels from images or raw time series measurements), you may want to use a neural network

For computer vision or NLP there's no doubt that the best models are specific types of neural networks.
However, despite a lot of work, they're not robustly interpretable (as can be seen from adversarial methods).
For tabular datasets it's generally a much closer game; and the idea of trying some different models and seeing what works best is often advocated, for example in Kuhn's [Applied Predictive Modelling](https://www.springer.com/gp/book/9781461468486).

There's often deeper advantages of interpretable models.
If the decision informed by the model is being made by a domain expert, they can understand why the model is making the prediction and use their expertise to override it if something is going wrong.
They're easier to maintain, understand and debug, easier to fine tune and hone, and unlikely to give surprisingly bad results.
All other things being equal and interpretable model is the best choice.

In the unstructured case you can use neural networks to extract structured features that then go into interpretable models.
I discuss this in [rules and models](/rules-and-models), and it's a standard approach in NLP using tools like [Stanza](/stanza) to extract the features and writing rules on top of those.

I'm not convinced by their claim that interpretable models can almost always be made to perform as well as black box models.
They don't occur in general competitive machine learning competitions (and didn't even win the interpretable machine learning competition).
Most of the research they point to is their own, and I don't know how strong the baselines are.
However I could believe they're good enough to use, and if a [simple model](/simple-model) does the trick to use it.

Cynthia Rudin has a lot of [interesting research](https://users.cs.duke.edu/~cynthia/papers.html) on this area.
There are papers on discrete optimisation methods such as branch-and-bound for interpretable methods such as [optimal sparse decision trees](https://arxiv.org/abs/2006.08690) ([code](https://github.com/Jimmy-Lin/GeneralizedOptimalSparseDecisionTrees)), [falling rule lists](https://arxiv.org/pdf/1710.02572.pdf) ([code](https://github.com/cfchen-duke/FRLOptimization)), and notably using linear [risk scores](https://jmlr.org/papers/v20/18-615.html) ([code](https://github.com/ustunb/risk-slim), [video](https://youtu.be/WQDVejk17Aw)).
There are prototype methods, in [image recognition](https://arxiv.org/abs/1806.10574) ([code](https://github.com/cfchen-duke/ProtoPNet)) including using [hierarchical prototypes](https://arxiv.org/abs/1906.10651) ([code](https://github.com/peterbhase/interpretable-image)), and in causal inference such as [MALTS](https://arxiv.org/abs/1811.07415) ([code](https://github.com/almost-matching-exactly/MALTS)) and [Almost Matching Exactly](https://arxiv.org/abs/1806.06802) ([code](https://github.com/almost-matching-exactly/DAME-FLAME-Python-Package)).
There are an assortment of other types of papers like [combining machine learning with decision making](https://arxiv.org/abs/1104.5061), [bandit methods in time series](https://arxiv.org/abs/1505.05629), and applied papers in healthcase, justice and electrical systems.
I'd like to dig deeper into a few of these later.