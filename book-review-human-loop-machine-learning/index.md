---
categories:
- books
date: '2022-07-30T08:00:00+10:00'
image: /images/human_loop_machine_learning.jpg
title: 'Human-in-the-Loop Machine Learning: Book review'
---

> Most machine learning models are guided by human examples, but most machine learning texts and courses focus only on the algorithms.
> You can often get state-of-the-art results with good data and simple algorithms, but you rarely get state-of-the-art results with the best algorithm build on bad data.
>
> Robert (Munro) Monarch, *Human-in-the Loop Machine Learning* 

[*Human-in-the Loop Machine Learning*](https://www.manning.com/books/human-in-the-loop-machine-learning) by Robert (Munro) Monarch is an excellent book on annotating data for Deep Learning practitioners in industry.
It covers the whole process from selecting data with active learning, to quality control for teams of annotators, to the annotation interface.
These are explained comprehensively focused on applying them in practice, and illustrated with common tasks such as text or image classification, image segmentation, object detection, named entity recognition, and translation.

I haven't been able to find many good resources on the holistic process of annotating data for machine learning, and this stands out as a gem.
Other books focus too much on the technical annotation model, representation, and tooling like [Natural Language Annotation for Machine Learning](https://www.oreilly.com/library/view/natural-language-annotation/9781449332693/) and [Training Data for Machine Learning](https://www.oreilly.com/library/view/training-data-for/9781492094517/) or dense academic tomes (like [Language Corpora Annotation and Processing](https://link.springer.com/book/10.1007/978-981-16-2960-0) or the [Handbook of Linguistic Annotation](https://link.springer.com/book/10.1007/978-94-024-0881-2)).
In contrast *Human-in-the Loop Machine Learning* is a readable, focused work on efficiently getting humans to annotate data for real applications, full of useful insights from the authors real world experience.
While at times it goes into great depth, for example in advanced active learning strategies, it's mostly material that's not well covered elsewhere, and the structure of the book makes it easy to only read what's relevant.
Given the multitude of resources available on machine learning it's really surprising there aren't more practical resources on data annotation.

The book consists of 5 parts; first steps, active learning, annotation, human-computer interface, and bringing it all together.
The first steps has a detailed hands-on example of annotating news headlines referring to a disaster illustrating many of the concepts of the book, and which I've [previously written about](/disaster-annotation).
Active learning explores ways of using models to effectively select data to annotate in a way that leads to better models with less human cost.
Annotation goes in depth in how to work with annotation teams from how to work with people to annotate data through to quality control.
Human Computer Interaction explores how to represent annotation tasks in a way that leads to fast, meaningful, and high quality annotation as well as exploring human-in-the loop use cases.
The final chapter brings everything together with 3 case studies of different human-in-the-loop machine learning models, with their source code.

# Active learning

> You should use active learning when you can annotate only a small fraction of your data and when random sampling will not cover the diversity of the data.

The active learning section discusses different ways to pick examples that are representative of a population, that the model is uncertain about, or that are diverse.
The recommended method is interleaving all three methods; the representative samples form a conservative baseline, the uncertain items help improve the model by identifying borderline cases, and the diversity sampling finds items the model is unlikely to have seen.
This also helps with human factors; uncertainty sampling and diversity sampling help pick interesting samples that keep the annotator engaged.

Representative sampling is the simplest method; most generally picking items completely at random from those available.
This helps the model get slowly better, but in many cases its inefficient because a lot of similar items end up being annotated that don't help the model.
It's always a safe fallback if the other methods fail.
There is some choice in terms of the population, and it can be extended by stratifying across different factors such as time or demographics to make it representative across these factors.

Choosing the examples that the model is uncertain about is a good way of getting the most model improvement per annotation.
It turns out there's a lot of ways of doing this and subtlety in the method.
In particular this book goes in depth into how the base (or equivalently temperature) of the softmax can change what ranks as "most uncertain"; this exemplifies what this book is great at - going into the details of things you won't find elsewhere.
It goes through increasingly sophisticated ways of active learning, such as using other models to predict the model uncertainty.

I hadn't previously considered the importance of diversity sampling; finding the "unknown unknowns" of the model.
However this is important in real world scenarios; we normally want the model to work in unusual cases.
It's not obvious how to find these; some methods suggested are using clustering, LDA, or samples where the last layer of the neural network has low activation.
Each of these methods can find different kinds of outliers, but the text gives no clear way to evaluate them.


# Annotation and accuracy

> The more respect you show the people who label your data, the better your data becomes.

The book has chapters on how to work with annotation teams, and how to evaluate their accuracy.
I haven't worked much in this area but the suggestions on annotation teams make a lot of sense, and seem to be drawn from the authors experience.
Treat the people who label data well, make sure they understand their impact, work closely with them.
There are good intuitive outlines of different commonly used statistics.

One concept I struggled with was the idea of "ground truth" and treating annotators as independent.
How do you get "ground truth" examples for data labelled by humans, when even experts will make mistakes?
Treating annotators as independent is statistically convenient for quality control, but might you get better results if you give feedback and let annotators collaborate?
While these ideas are useful for modelling quality control, some careful thought needs to go into any particular usecase.

> A real-time chat allows annotators to collaborate on hard-to-annotate items, the only downside being that quality control becomes harder when annotators aren't independent.

Indeed what "ground truth" is depends a lot on the application.
Designing an appropriate annotation model and guidelines is slippery business, and there are lots of edge cases.
The book could have gone into more detail here about choosing and defining tasks appropriate to a business goal and dealing with ambiguity.

> we spent more time on refining the definiiton of what goes into a span than on any other part of the task

There are some good references from the book that go into more of the difficulty of annotating data:

* [Truth Is a Lie: Crowd Truth and the Seven Myths of Human Annotation](https://ojs.aaai.org//index.php/aimagazine/article/view/2564)
* [Are We Modeling the Task or the Annotator? An Investigation of Annotator Bias in Natural Language Understanding Datasets](https://arxiv.org/abs/1908.07898)
* [Revolt: Collaborative Crowdsourcing for Labeling Machine Learning Datasets](https://www.microsoft.com/en-us/research/publication/revolt-collaborative-crowdsourcing-labeling-machine-learning-datasets/)


# Interfaces

The user interface can make a huge impact on efficiency, accuracy, and the agency in annotation.
There are lots of good examples in the text about designing user interfaces, such as using predictive typing to speed up translation (which is much more efficient than correcting suggestions), or instead of highlighting sentiment words asking to edit the words to express the opposite sentiment (changing the task to a more informative one), or [Extreme clicking for efficient object annotation](https://arxiv.org/abs/1708.02750) where instead of drawing a bounding box you click on each of the highest, lowest, left-most and right-most point (which is 6x more efficient, and captures more information).

There's lots of general user experience tips in here too.
Making sure the interface can capture all the information annotators feel are important (general features include being able to add notes, or flag examples).
If end users are annotating an optional field, a suggestion is to pre-fill it because "people are more averse to wrong data than to missing data" and are more likely to correct it.

> The biggest shortcoming of training data provided by your users is that the users essentially drive your sampling strategy.

# Conclusion

> Understanding the human task that you are solving will help every aspect of your prdoct design: interface, annotations, and machine learning architecture

The last chapter brings everything together in three fully coded Human-in-the Loop examples on [Bicycle Detection](https://github.com/rmunro/bicycle_detection), [Food Safety](https://github.com/rmunro/food_safety) and [News Headlines](https://github.com/rmunro/headlines).
I'll go into more detail on them in following posts, but these are very interesting and exciting examples of how these techniques can be used.

These kinds of tools could completely change the face of analytics.
Working as an analyst machine translation (e.g. [including via OpenNMT and Marian MT](/python-offline-translation)) has greatly increased my ability to find information in documents in other languages.
However this just scratches the surface; these kinds of tools can help extract concepts via example and quickly scale annotation.
I've previously found analytics has focused on things easy to extract; things that match regular expressions (such as phone numbers or email), or keyword matching with stemming and splitting.
Anything that requires custom development is too expensive for a lot of one-off analytics use cases, but these techniques could greatly reduce the cost.

They have already changed the face of digital products.
Things like auto-completion, pre-filling fields, and scrolling feeds are increasingly powered by machine learning with tight feedback loops.
They involve a tight coupling between a computer interface and a machine learning model, that helps people achieve a task and allows appropriate annotation (often implicitly or explicitly by the end user).
Some good resources referenced in the book for these are Microsoft's [Guidelines for Human-AI Interaction](http://erichorvitz.com/Guidelines_Human_AI_Interaction.pdf) (see also their [Human-AI eXperience (HAX) Toolkit](https://www.microsoft.com/en-us/haxtoolkit/)) and [Building Machine Learning Powered Applications](https://www.oreilly.com/library/view/building-machine-learning/9781492045106/) by Emmanuel Ameisen.

This book fills a major gap in all the machine learning resources I can find, in a practical guide of how to efficiently collect accurate labelled data with worked examples of Human-in-the Loop processes.
It focuses much more on annotation being a *human* task to help *human* ends than other machine learning books, which either take it as a given or focus on the technical process.
It's surprising there's not more of this; when I studied Physics we learned about our measurement equipment, the basics of electronics, the kinds of systematic and random errors that occur, and did laboratory work of running experiments.
I haven't seen another machine learning book that talks about human factors in annotation, the basics of psychology, the kinds of systematic errors (such as priming) that occur, and actually labelled some data.
The closest I've seen is in survey methodology, which tends to be focused more in marketing and user experience departments, and is different to many of the annotation tasks in machine learning.
Hopefully as transfer learning and better tooling makes machine learning more accessible, there will be more resources like this book to ensure we get the most out of them.

> Rather than spending a month figuring out an unsupervised machine learning problem, just label some data for a week and train a classifier.
> 
> [Richard Socher](https://twitter.com/richardsocher/status/840333380130553856?lang=en)