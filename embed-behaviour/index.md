---
categories:
- data
date: '2020-12-02T21:34:14+11:00'
image: /images/network_graph.png
title: Using Behaviour to Understand Items
---

When people access products online their behaviour gives lots of information about both the people and the products.
This information deeply enriches understanding of how to better serve your customers, how your products are related to each other and can help answer deeper questions about them.
However you need to find a way to unlock the information.

Using behavioural information can greatly improve modelling on the tabular data in your database.
In natural language processing there a large number of tasks like predicting whether a sentence is saying something good or bad ([sentiment analysis](https://paperswithcode.com/task/sentiment-analysis)), identifying people and places in a document ([Named Entity Recognition](https://paperswithcode.com/task/named-entity-recognition-ner)) and translating text from one language to another ([machine translation](https://paperswithcode.com/task/machine-translation)).
In recent years performance on all these tasks have increased dramatically by firt [pretraining a language model](/dont-stop-pretraining) before training on the specific task.
A language model predicts the next word (or letters) that will come part way through a sentence, like the next word suggestions on your phone's keyboard (or [write with transformer](https://transformer.huggingface.co/doc/gpt2-large) to see a state of the art example).
By becoming really good at predicting the next word a language model gets really good representations of language, which allows it to better determine what a sentence is saying, whether these words are someone's name and how to produce grammatical output in a translation.
Recently the same idea has started to show promising results in computer vision, where it is called [self-supervised learning](https://www.fast.ai/2020/01/13/self_supervised/).
So if you've got a large database filled with behavioural information then you're only going to get the best results on building on that knowledge.

Creating tasks that embed categorical items is a powerful way to use behavioural data, and enables use of the embeddings in other tasks.
I've already written about [building categorical embeddings](/embeddings); in short instead of treating items as independent it lets you put them as points in some vector space where similar items are close together.
Suppose you're trying to gauge interest on some very niche category of item; you don't have much direct information on the category and so can't directly measure it.
But in an embedding you can use information about interest of related items to infer how interest is changing for this item.

Thinking about a website like eBay there are lots of different ways you can use behaviour to understand items.
Uou could use people's intent behaviour on items to understand both of them.
For example people who view, watch (save), and buy many of the same items are likely similar in some way and products that are viewed, saved and bought by many of the same people are similar.
But this doesn't work for all sorts of purchases; maybe you need to filter it by search behaviour.
Items that are likely to be clicked on by the same search term are likely to be similar in a more contextual sense.
Items that are often posted together by the same person are similar in a different sense.
Items that contain similar text descriptions or a similar photograph are similar in yet another way.
You could do this for individual items, categories of items, or aspects of items.
You could even use this [to make categories of items](/cluster-exploration).

It would be great to demonstrate this effect in the open, but it's hard to find good datasets.
A good real example of this is [the 3rd place solution to the Kaggle *Rossman Stores Sales Data* competition](https://github.com/entron/entity-embedding-rossmann), where they solved it with a neural network and got informative embeddings of locations.
The [Google Analytics sample dataset](https://console.cloud.google.com/marketplace/product/obfuscated-ga360-data/obfuscated-ga360-data) is a good candidate, containing user behaviour in the Google Merchandise store.
There's an [ecommerce behavioural dataset on Kaggle](https://www.kaggle.com/mkechinov/ecommerce-behavior-data-from-multi-category-store) which also could be good.
Or using a big public forum like [pushshift.io's export of Reddit](https://archive.org/details/2015_reddit_comments_corpus), [Hacker News](https://console.cloud.google.com/marketplace/product/y-combinator/hacker-news) or [Stack Exchange](https://archive.org/details/stackexchange) could be used by looking at people that comment in the same threads.
In each of these cases it would be worth nominating a task to show improvement on, like [predicting revenue form Google Analytics](https://www.kaggle.com/c/ga-customer-revenue-prediction) or [closed questions on Stack Overflow](https://www.kaggle.com/c/pycon-2015-tutorial-predict-closed-questions-on-stack-overflow).