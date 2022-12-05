---
categories:
- python
- nlp
date: '2020-07-09T08:00:00+10:00'
image: /images/stanza.png
title: Stanza for NLP
---

Working with unstructured text is much easier if we add structure to it.
[Stanza](https://stanfordnlp.github.io/stanza/) is a state of the art library for doing this in over 60 languages.
Given some text it will tokenize, sentencize, tag parts of speech and morphological features, parse syntactic dependencies and in a few languages perform NER.
It's easy to use and gets extremely good results on benchmarks for each of these tasks on a large number of languages.
The only drawback is it's slower than some simpler models, but it's fast enough to use in production.

It's quite easy to use:

```python
import stanza
stanza.download('en')
nlp = stanza.Pipeline('en')
doc = nlp('Edward thinks Stanza is really neat in July 2020')
print(doc.entities)
```

Which prints out the detected entities

```
[
{"text": "Edward", "type": "PERSON", "start_char": 0, "end_char": 6},
{"text": "Stanza", "type": "PERSON", "start_char": 14, "end_char": 20},
{"text": "July 2020", "type": "DATE", "start_char": 39, "end_char": 48}
]
```

We can look at individual tokens; e.g. `doc.sentences[0].tokens[1]`.
It gives not only the part of speech, lemma and dependency relation, but also a list of features from [Universal Features](https://universaldependencies.org/u/feat/index.html).
The word "likes" is singular form, 3rd person, present tense, indicative mood and a finite verb form.
My linguistics isn't good enough to understand all of this, but I don't know another way to determine [singular versus plural form](/making-words-singular) so it seems useful to me.

```
[
  {
    "id": "2",
    "text": "thinks",
    "lemma": "think",
    "upos": "VERB",
    "xpos": "VBZ",
    "feats": "Mood=Ind|Number=Sing|Person=3|Tense=Pres|VerbForm=Fin",
    "head": 0,
    "deprel": "root",
    "misc": "start_char=7|end_char=13"
  }
]
```

Stanza was introduced in an [ACL 2020 demo](https://www.aclweb.org/anthology/2020.acl-demos.14.pdf).
It's a production version of the neural pipeline described in [Universal Dependency Parsing from Scratch](https://arxiv.org/pdf/1901.10457.pdf).
I would like to dig more into this later, including [Deep Biaffine Attention for Neural Dependency Parsing](https://arxiv.org/pdf/1611.01734.pdf).

More traditional pipelines such as spaCy 2 use rules for tokenization and lemmatisation.
By using neural networks it means to extend Stanza to another language you just need enough annotated data; which are easier to obtain than linguistic expertise to craft rules.
In practice there's now enough data and good enough techniques that neural methods are more accurate than rules. 
The main drawback is that the rules are slower to train and run.
According to their benchmarks going from raw text to dependencies is about 10x slower than spaCy (but only 3x on a GPU).

The NER algorithm is based on [Contextual String Embeddings for Sequence Labeling](https://www.aclweb.org/anthology/C18-1139.pdf) used in [flair](https://github.com/flairNLP/flair).
The NER performance on benchmarks is similar to flair, but it runs much faster.
I'm really interested in the idea of using it to train custom NER models.

It's also straightforward to use in spaCy with [spaCy-Stanza](https://github.com/explosion/spacy-stanza) by wrapping it in a `StanzaLanguage`.
This means it can be used with spaCy's matcher and displacy tools for convenience.

I still need to experiment with it more but it looks like a *very* promising library for NLP.