---
categories:
- nlp
- sentiment
date: '2023-04-14T19:00:00+10:00'
image: feeling_wheel.png
title: Learning Natural Language Processing through Sentiment Classification
---

There is currently a lot of hype in Natural Language Processing, much of it driven by Open AI's effective marketing of its GPT systems.
As someone who has only been working in NLP for a few years, I find it hard to understand what systems and techniques are available, what their capabilities are, and their limitations.
To get a solid grounding in these methods I want to go really deeply understand one small problem - in this case Sentiment Classification.

A few years ago I looked at promising resources for [NLP available in 2020](/nlp-resources-2020), and they've provided me a useful foundation but I still don't know enough to debug what to do when a model doesn't train as well as I expect.
[Stanford CS224N](https://web.stanford.edu/class/cs224n/) ([videos](https://www.youtube.com/playlist?list=PLoROMvodv4rOSH4v6133s9LFPRHjEmbmJ)) is great for understanding how the field evolved and important components, but the lectures and [assignments](http://web.stanford.edu/class/cs224n/assignments/) follow the happy path of everything going right.
However as soon as something goes wrong with training in practice I don't know how to handle it, and I don't know how to improve it.
[Stanford CS224U](https://web.stanford.edu/class/cs224u/) ([videos](https://www.youtube.com/playlist?list=PLoROMvodv4rPt5D0zs3YhbWSZA8Q_DyiJ)) which is fantastic at teaching how to understand the problems, techniques and datasets in NLP, with [challenging assignments](https://github.com/cgpotts/cs224u) with an open bake-off component that let me experiment with different approaches, but it (rightly) focused on breadth over depth.
The [fast.ai course](https://course.fast.ai/) and [book](https://github.com/fastai/fastbook) are really good at getting into the detail, and I find the technique of breaking apart a system top down and building it back bottom up is really useful; I learned a lot [digging into fastai with Fashion MNIST](/peeling-fastai-layered-api-with-fashion-mnist) and then [rebuilding it](/building-layered-api-with-fashion-mnist), but the NLP section is very limited and doesn't touch modern Transformer methods.
The [Natural Language Processing with Transformers book](https://www.oreilly.com/library/view/natural-language-processing/9781098136789/) is great at showcasing how to use the HuggingFace transformers library for applications (like [building a recipe ingredient classifier](/recipe-ner-transformers)), but only touches on the details of how it all hangs together.

The problem is I don't have strong enough mental models of how these things models work to be comfortable with them.
I hear that RNNs can't store long term dependencies because of vanishing gradients, are inefficient because of their sequential nature, and are unstable because of exploding gradients; yet somehow [RWKV-LM](https://github.com/BlinkDL/RWKV-LM) on the surface appears to be an efficient language model.
There's a lot of talk about few-shot tuning and prompt engineering; I have an intuition that if you have a lot of good data fine-tuning a small model will get a better result, but I don't have an easy way to test that.
When I try to train a large model and it doesn't converge as it did in the paper I don't know how to diagnose it and start guessing; maybe learning rate or batch size?
It seems [feasible to pretrain a lanugage model from scratch](/why-train-language-model) - why don't I hear people doing it much?

To understand the history, and hopefully the future, requires taking a deep thin slice; a good example is [the history of Machine Translation](https://en.wikipedia.org/wiki/History_of_machine_translation).
Machine translation is a complex topic and I'll likely get the details wrong, but here's a plausible narrative.
Prior to computers most machine translation was done by people who could understand both languages and the content of the text; learning another language takes significant time and effort (although less in specialised domains such as academic texts), and the main constraint was human translator expertise.
The first wave of Artificial Intelligence in the 1950s to the 1980s was based on rules, logic, and search; for [rule based machine translation](https://en.wikipedia.org/wiki/Rule-based_machine_translation) this involved analysing the morphology of the words in the source sentence, parsing the sentence structure, mapping the words with a bilingual dictionary and generating appropriate morphology, then un-parsing them into the target language.
This worked very well on small examples, but on large tasks to translate real texts performed very poorly because of ambiguity, the size of the lexicon, and the complexity of grammar and language in real use (there are some good examples in the [ALPAC report](https://web.archive.org/web/20110409070141/http://www.mt-archive.info/ALPAC-1966.pdf) of the state of the art in 1966) - they just weren't good enough for widespread practical use.
It did have some real use cases in narrow domains; from 1978 Xerox used SYSTRAN to translate technical documentation to many lanugages, and by restricting themselves to writing using a simplified language and grammar it allowed them to scale to many languages with minor post-editing from professional translators.
The primary constraints were the human linguistic expertise in building and maintaining dictionaries, morphological analysers, and parsers that could handle the variety of real language.
With the exponential rise of more storage, computing power, and data, in the 1980s and 1990s more data-driven approaches appeared, namely Example Based Machine Translation and Statistical Machine Translation.
[Example Based Machine Translation](https://en.wikipedia.org/wiki/Example-based_machine_translation) replaces the bilingual dictionary with looking up phrases in a database of parallel texts, and heuristically aligning them, and along this line using data driven methods to create resources like dictionaries became more common.
[Statistical machine translation](https://en.wikipedia.org/wiki/Statistical_machine_translation) techniques build a probalistic model from bilingual data, and try to find the most likely translation.
This involves using parallel corpora to learn phrases (using a seprate model to first align the phrases from the text), and then a target n-gram language model to generate likely text.
These were the first systems that were good enough for use on general text (including the early versions of Altavista's Babelfish and Google Translate); while they often had errors and were not fluent in the target language they were useful enough to understand for high resource languages.
Alignment of words and phrases is a difficult problem, and errors would cascade into the machine translation system, making it a weak link in the chain.
More fundamentally the models were sparse, this meant unusual phrases, errors, and idioms would be poorly translated, and the n-gram lanugage models would have poor long range coherence.
The main limitation was data; making a system better required exponentially more aligned data for translation and monolingual data for language modelling.
There were some systems that could generalise better using syntax translation, but this was another potential weak link in the chain that could propagate errors.
In the mid-2010s [neural machine translation](https://en.wikipedia.org/wiki/Neural_machine_translation) overtook statistical machine translation, by using *dense* representations of tokens in a model with more parameters allowing better generalisation, and learning end-to-end (without the separate alignment step).
The constraint is primarily compute and secondarily data; for low-resource languages data augmentation techniques like back-translation can help increase effectiveness and for generation larger models generally get much more fluent.
Neural translation is now *really* good, I'm sure there are still more gains to be hard by learning more efficiently with compute or data but it's unclear what they are.

The history of machine translation covers many of the typical changes in Natural Lanugage Processing.
The earliest systems operated in an environment when compute and disk space was very expensive, and computer expertise and availability were rare, and tended to be rule based on small lexicons.
Around the 1980s it started to be practical to infer some of these rules from data.
In the 1990s to 2000s it became feasible to build prototype and statistical models from labelled datasets, but getting good data efficiency required hand-crafting features.
In the mid-2010s increasingly cheap compute made it possible to learn dense representations which, with a large enough labelled dataset, could learn features automatically.
In the late-2010s transfer learning, in particular language modelling, meant that models could be learned with much less data.
And now we're at a point where very large pre-trained models can do well in the zero or few-shot setting.
How true is this portrait, and how much better are these models in a production setting where we have to worry about speed and debugging, as well as accuracy and development velocity?

I think the best way to progress on this is to go really deep on one specific task.
Andrej Karpathy's [Deep Neural Nets: 33 years ago and 33 years from now](karpathy.github.io/2022/03/14/lecun1989/) shows how taking an old technique and casting it through a modern lens can be informative for looking forward.
A good task should:

* be general enough to be informative
* have public available datasets
* famous enough there are many good benchmarks to compare with for many techniques
* general enough that modern techniques can be applied to it
* large enough that improvements can be distinguished from noise
* large enough for deep learning methods
* hard enough that it doesn't saturate too quickly with better models
* interesting enough to stare at it for a couple of months

It's hard to meet all these criteria but *sentiment classification* is a strong candidate, comsidering a few datasets together.
It's a specific case of text classification, the general task of taking unstructured text and breaking them into groups according to some criteria, that is generally useful.
The datasets I would initially consider are:

1. The Stanford Sentiment Treebank [Socher et al., 2013](https://aclanthology.org/D13-1170.pdf) is 11,855 sentences with fine-grained sentence annotations at the phrase level. It extends back to classical models, being built from the movie review dataset of [Pang and Lee, 2005](https://aclanthology.org/P05-1015.pdf), but extends into current models and is included in common benchmarks like GLUE and SentEval.
2. [Maas et al., 2011](https://aclanthology.org/P11-1015.pdf) [released](https://ai.stanford.edu/~amaas/data/sentiment/) a 50 thousand sentence IMDB dataset with binary labels (rating below 4 is negative, above 7 is positive), which is used in the period from machine learning on dense representations through to today in benchmarks such as MTEB. [Gardner et al., 2020](https://aclanthology.org/2020.findings-emnlp.117.pdf) [released a contrast set](https://allenai.org/data/contrast-sets), and [Kaushik et al., 2019](https://openreview.net/forum?id=Sklgs0NFvr) released counterfactually updated data which can help uncover overfitting.
3. Yelp releases several review datasets with star ratings, and in particular [Zhang, Zhao, and LeCun, 2015](https://arxiv.org/abs/1509.01626) released a [version of the Yelp dataset](https://github.com/zhangxiangxiao/Crepe) that has been used in many comparisons.
3. [Potts et al., 2020](https://arxiv.org/pdf/2012.15349.pdf) released DynaSent which is based on Yelp restaurant reviews and is adversarially constructed against deep learning models, and should be robust against harder models.

These are an interesting set for comparison of transfer learning within domain, and across domain, especially when coupled with unlabelled corpora.
A potential extension would be into Natural Lanugage Inference datasets such as SNLI, MNLI, and ANLI for studying transfer across tasks.

# Work plan

I've got a rough idea of what I want to do, although it will likely change after the literature review.
The overall structure loosely follows the first part of [Jurafsky and Martin's Speech and Lanuage Processing](https://web.stanford.edu/~jurafsky/slp3/).

1. Literature review
1. Initial analysis of the datasets and manual annotation
3. Evaluation Plan
4. Manual Features
5. Lexicons and feautre mining
6. N-gram lanugage models
7. Sparse Feature Models
8. Dense Feature Models
9. Neural Models
10. Transformers

I'm sure this will change as I get deeper into the work, but I would like to have touched on most of these points.
