---
categories:
- data
- nlp
date: '2020-07-12T08:00:00+10:00'
image: /images/nlp_checklist.png
title: A Checklist for NLP models
---

When training machine learning models typically you get a training dataset for fitting the model and a test dataset for evaluating the model (on small datasets techniques like cross-validation are common).
You typically assume the performance on your chosen metric on the test dataset is the best way of judging the model.
However it's really easy for systematic biases or leakage to creep into the datasets, meaning that your evaluation will differ significantly to real world usage.
In the worst case your model could be giving really odd results decreasing the impact of your machine learning product.

The tool [Checklist](https://github.com/marcotcr/checklist), presented [in an ACL 2020 paper](https://www.aclweb.org/anthology/2020.acl-main.442.pdf), presents a way to get more assurance on an model with language input.
The suggest 3 *types* of tests:

1. Minimum Functionality Tests: Basic things the model should do (e.g. "This was a fantastic meal" should be positive sentiment)
2. Invariance Tests: Specifying changes to input that shouldn't change the output (e.g. "This was a fantastic meal" and "This was a fantastic hotel" should both have the same sentiment)
3. Directional Tests: Specifying changes to the input that should change it just one direction (e.g. if we add "The deserts were excellent" it should not decrease the sentiment).

I gave the examples in terms of sentiment, but it really works for any model with a cateogorical or quantitative output.
They introduce a wide range of capabilities to perform these types of tests on to get a matrix.
They gave examples like vocabulary (e.g. changing words), robustness (e.g. simulating spelling mistakes), NER (changing names), negation, coreference resolution, semantic role labelling and logic.
Some of these are quite exotic but it's a good way of coming up with a wide range of examples.

They provide a lot of tooling to help generate tests, as well as rewrite tests in a fluent way for invariance and directional tests, but the main thing I got out of this is the idea of actually *writing* data tests.
Simply having and running tests like "this simple text should almost certainly to this category" (with some probability) is a very good failsafe for a model.
It doesn't replace metrics, but it can catch serious errors in the dataset and help show things not captured in the headline metrics.

From now on I'm going to specify at least minimum functionality tests for any model I build.
I need to look into their tool in more detail to see whether it is worth using.