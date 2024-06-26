---
categories:
- data
date: '2020-09-01T08:00:00+10:00'
image: /images/embedding.png
title: Embeddings for categories
---

Categorical objects with a large number of categories are quite problematic for modelling.
While many models can work with them it's really hard to learn parameters across many categories without doing a lot of work to get extra features.
If you've got a related dataset containing these categories you may be able to meaningfully *embed* them in a low dimensional vector space which many models can handle.

Categorical objects occur all the time in business settings; products, customers, groupings and of course words.
For models like linear or logistic regression or nearest neighbours the way to deal with these is by one-hot encoding.
This means you end up with as many parameters as there are categories, and this can quickly become unwieldy.
You end up having to do a bunch of preprocessing to normalise similar things, maybe even building custom hierarchies, and throwing away infrequent items because you don't have enough data.
In some cases this can work quite well as a starting point; but you lose the long tail which often has a lot of interesting information.

Tree based methods like gradient boosting trees and random forests can directly handle categorical data.
However because they split the categories at cut points the *order* you give the categories in makes a big difference; close together items will tend to be grouped together.
If you can group similar categories together you can expect much better performance, but it's not always obvious what similar means and how to do this.
While you will get the tail data in your model you can't rely on it.

Embeddings use information you have to map the categorical items in a vector space where similar items are close together.
This is done by building a model based on the structure of the information you have about the items.
You can think of it as a lookup array; each category is mapped to a vector.
What *similar* means depends on the information you use.

For text a classic approach is [Word2Vec](https://en.wikipedia.org/wiki/Word2vec) where each word in the corpus is mapped originally to random vectors.
Then a pair of shallow neural networks are trained to predict neighbouring words.
The embeddings end up where words that could be used in the same context end up close by in the vector space.
There are many variations on this method, but training a language model on a lot of text is still generally the best way to get good models for solving other language tasks.

For products and customers a typical approach is to use the interaction between products and customers.
Customers who purchase the same products are in some sense similar; products that are purchased together are in some sense similar.
You could then factor this interaction matrix to get an embedding of users and an embedding of customers.
Another method would be to construct the [product-product matrix](/recommendation-graph) and then consider this as a distance matrix which you embed with [multidimensional scaling](https://en.wikipedia.org/wiki/Multidimensional_scaling).

These techniques are used in industry, for example [Pinterest](https://medium.com/the-graph/applying-deep-learning-to-related-pins-a6fee3c92f5e) and [Instacart](https://tech.instacart.com/deep-learning-with-emojis-not-math-660ba1ad6cdc).

One thing to keep in mind is that similarity will be defined by the type of data that you model.
For example if you tried to predict items that would be purchased in the same session then similar items would be *complementary* items; things that go well together.
When you try to predict items that are purchased by similar customers then you are more likely to get items that are *substitutes* close together.
If you predict items that are viewed you're also going to pick up some *aspirational* items; a lot of people like window shopping for objects they won't actually buy.
Consider the kind of similarity that would be useful to capture for downstream models.

It would be interesting to get some open data and demonstrate how well some simple product embedding approaches work for improving related modelling tasks.