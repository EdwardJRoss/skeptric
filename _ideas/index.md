---
categories:
- ''
date: '2020-03-23T08:00:27+11:00'
draft: true
title: _Ideas
---

# Sleep

Importance of
Everything is worse on little sleep (parents)
Learning
Longevity

# Stimulants

## Caffine
Sensitivity

Grumpiness, jitteriness, interruptiness

Sources: Coffeee, dark chocolate, tea

### Coffee
Pollan: https://www.theatlantic.com/magazine/archive/2020/04/michael-pollan-coffee/606805/
Balzac: http://blissbat.net/balzac.html

Many forms (espresso, milky, filter/french press, italian, turkish/arabic)

People defining themselves with their order

Needing more - ramp up
Culture; workplace coffee machines, catching up with people
Alternatives; decaf (not available), tea/chai, juice, water

Carsales coffee machine headache

### Withdrawal

Headaches


## Exercise

## sugar

## Hot showers

## Chilli

## Fast music

## Social stimulation

# fastText file format

JS implementation?

# Optimising the wrong metric

Proxy metrics

Arrests
Youtube -> Clickbait

# Histogram optimisation

# Sampling in shell with shuf

shuf data | \
head -n ${SAMPLE_SIZE} | \
cut -d$'\t' -f1 | \
sort | \
uniq -c | \
sort -nr | \
sed -e 's/^ *//' -e 's/ /,/' | \
awk -F',' -v total="$SAMPLE_SIZE" 'BEGIN {OFS=","} {cumsum+=$1; print $1/total * 100 "%", cumsum/total * 100 "%", $2}' |\
head

# Group by with shell

a,b,sum

# Comm

# SRI Model

# Modelling with limited or bad data

# Association rules

# Blitzmail

# website categorisation

# hadoop streaming

Benefits of the cloud - resource constraints

# Git submodules

When should you use a submodule?


# Lua

A nice small language with some weird ends
Counting with `#`
There's a list somewhere

# Scheme

A nice small language with some interesting ends. Call/CC, macros

SICP

SICM

# Pseudorandom numbers

# Art of Insight

# Graphing and visualisation

# Deploying a model

# J programming language

# Using search to solve problems

Minikanren

Optimisation

# Optimisation vs Model training

High information scenarios

# Next best offer

# Clustering tex

SVM

CART

Boosting and bagging

ARIMA

Neural nets


# Running open source phone

andOTP

AntennaPod
Voice
Music

Blitzmail

F-Droid

Fennec F-Droid

K-9 Mail

KeePAssDroid

Kore

OpenVPN for Android

OsmAnd~
UnifiedNlp

primitive ftpd

Soft sound

Syncthing


# Recall and precision in information retrieval
The nomenclature is confusing: https://en.wikipedia.org/wiki/Evaluation_measures_(information_retrieval)#Precision
Note that the meaning and usage of "precision" in the field of information retrieval differs from the definition of accuracy and precision within other branches of science and statistics.


# Mail syncing on mobile phone

# PII is the wrong abstraction

# Define your work

# What I learned from honours

# Connecting to redshift with psql

# User aggregation with hyperloglog


# Flinders street foot counters since lockdown


# caniuse for databases

Or maybe a rosetta table?

# CTE

# Testing if two SQL tables are equal

# Formatting SQL

# Bad words with SQL split

# Cum pct with SQL

# SQL Assertions

great expectations

# Using images in blog posts

# SQL Time since last visit

# SQL New/Return/Churn

# Dbplyr dates

# Improvement

- Add fiddles to code?
- Add titles to tables
- Add syntax highlighting

# Experiments for burn-in/seasonal: How general are they?

Christmas Button
COVID-19
Novelty effect



# Generalizability of experiments
People take A/B as gold but how *generalizable* is it?

COVID-19: A different population
Christmas button: A different environment

# p-values

# R for data science

# ggformula

# Central limit theorem for skew distributions

# log-linear curve fitting

Fitting in log is different to fitting logarithmic function

# barry esseen inequality

# Why a logit function?

# KDE

# How good is open source

# Sacha Chua - Blogging

https://sachachua.com

https://gumroad.com/l/no-excuses-blogging

Write about what you're learning

> Write early, write often. Don't wait until you've figured everything out.

> You're probably learning something new every day.

> The easiest blog post to write is the answer.

> Share while you learn

There's something to share everywhere through the process

> Don't worry about your strategy.

> When you start you'll be boring

It's okay to write about different things

> Turn your ideas into small questions, and then answer those.

> > Break that question down into smaller questions until you can actually answer it in one sitting.

> Make Sharing part of the way you work

# Hide-show mode

# Tqdm

# Daily habits

# Mittens
https://github.com/roamanalytics/mittens

# SpaCy Dependency Matcher

http://markneumann.xyz/blog/dependency_matcher/


# Document clustering Paper

https://aircconline.com/mlaij/V3N1/3116mlaij03.pdf

# Kubeflow

Pros and cons

# MLFlow

# Kedro

# Metaflow

# Difference of two confidence intervals is different to confidence interval of difference

# RWalkr and suggrants and covid19


# Packrat
https://rstudio.github.io/packrat/

# CI/CD Hugo

# Post data

```
grep -L '^draft' *.mmark *.Rmd  | xargs -n1 grep -H '^date' | sed -E 's/^([^:]*).*([0-9]{4}-[0-9]{2}-[0-9]{2}).*/\2,\1/' | sort
```

Pragmatic

```
grep -L '^draft' *.mmark *.Rmd  | xargs -n1 grep -H '^date' | sed -E 's/^([^:]*).*(2020-0[3-9]-[0-9]{2}).*/\2,\1/' | grep '^2020' | wc -l
```

```
grep '^tags' * | sed 's/.*\[//' | sed 's/[]" ]//g' | tr ',' '\n' |  sort | uniq -c | sort -nr
```

# Pylint

https://pythonspeed.com/articles/pylint/

--score=n
--reports=n

Disable

        [R]efactor for a “good practice” metric violation
        [C]onvention for coding standard violation
        [W]arning for stylistic problems, or minor programming issues
        [E]rror for important programming issues (i.e. most probably bug)
        [F]atal for errors which prevented further processing


# Python CLI

https://changelog.com/posts/pynt-versus-invoke

https://github.com/micheles/plac

https://github.com/google/python-fire

https://github.com/docopt/docopt

https://click.palletsprojects.com/en/7.x/

https://github.com/tiangolo/typer

fastcli

# Which key

# Delegation

Do, Defer, Delete, Delegate

Delegating up and accross

# Operation

# Pandas sort: kind=mergesort for stable

https://stackoverflow.com/questions/17141558/how-to-sort-a-dataframe-in-python-pandas-by-two-or-more-columns

https://stackoverflow.com/questions/14941366/pandas-sort-by-group-aggregate-and-column/14946246#14946246

# Teach

```python
import spacy
import pandas as pd
import json
df = pd.read_feather('../data/04_secondary/ads.feather')

nlp = spacy.load('en_core_web_lg')

sample = df.sample(1000).FullDescription.str.replace('****', 'REDACTED', regex=False).replace('\u2022', ' . ', regex=False)

f = open('job_ad_sentences.jsonl', 'w')

docs = nlp.pipe(sample)

for doc in tqdm(docs, total=len(sample)):
  for sent in doc.sents:
    json.dump({'text': str(sent)}, f)
    f.write('\n')

f.close()
```

```
prodigy textcat.teach -l skill sentence_skill ./blank_model/ ./job_ad_sentences.jsonl
```

# Tech since the late 90s

## Wikipedia 

Britanica, Encarta

Wiktionary


## Scripting: Python, Perl and Javascript

Python ecosystem (Numpy, Matplotlib, Scipy)

R

## Dotcom crash

Banner ads

Rise of Advertising and Eccomerce

## Difficulty of making websites

## Microsoft vs Netscape

Best viewed with IE

Mozilla

Google is the new Microsoft

## Flash

Adobe turned it into tracking nightmare

Nothing today seems as easy to make multimedia


## Lack of interactivity

Everything HTML, simpler

AJAX (too many SPA)

## Linux

Linux: 1991

Microsoft anti free software



## XML 

Astronaut Architecture

XML 1.0: 1998

XSLT (1998-99), DSD

XHTML
SOAP

(Good: SVG, RSS, Atom, Office XML!)

Reaction to binary formats

Ant -> Maven

Now: JSON, YAML (YAML templating!)

## Unicode

Very good.
Edges: Han Unification, UTF-16 & 32.

## Java


# Finding job data

https://dmorgan.info/posts/common-crawl-python


# Github actions cache 

https://help.github.com/en/actions/configuring-and-managing-workflows/caching-dependencies-to-speed-up-workflows


# RDF



## RDF Graph to Dictionary

#### Getting relevant nodes

```
t = rdflib.term.URIRef('http://www.w3.org/1999/02/22-rdf-syntax-ns#type')
jp = rdflib.term.URIRef('http://schema.org/JobPosting')
def get_job_postings(G):
    return G.subjects(t, jp)
```

```
list(G.subject_objects(t))
```

```
nodes = set(s for s,p,o in G if type(s) == rdflib.term.BNode)
referred_nodes = set(o for s,p,o in G if type(o) == rdflib.term.BNode)
root_nodes = nodes - referred_nodes
```


#### Graph to Dictionary

```
def graph_to_dict(G, j, seen=frozenset()):
    d = {}
    for k, v in G.predicate_objects(j):
        k = k.toPython()
        if v in seen:
            logging.warning('Detected cycle in RDF %s', G.identifier)
            return None
        elif type(v) == rdflib.term.BNode:
            v = graph_to_dict(G, v, seen.union([v]))
        else:
            v = v.toPython()
        d[k] = d.get(k, []) + [v]
    return d
```

# Community Detection

Reichardt and Bornholdt - Statistical Mechanics of Community Detection - https://arxiv.org/pdf/cond-mat/0603718.pdf

Defines Hamiltonian spin glasses; effectively same metric as below: `\sum_{C}\sum_{i,j \in C} tA_ij - \frac{k_i k_j}{2m} `


Lambiotte, Delvenne, and Barahona - Laplacian Dynamics and Multiscale Modular Structure in Networks - https://arxiv.org/pdf/0812.1770.pdf

Defines it as a linearisation of independent, identical homogeneous Poisson process (dynamics of normalised Laplacian matrix).
Have code https://xn.unamur.be/codes.html (timescale.zip)
Explains implementation in https://xn.unamur.be/codes/readme.pdf
Built on Louvain method for optimising modularity
Reimplemented in https://github.com/taynaud/python-louvain


Louvain method: Fast unfolding of communities in large networks
Vincent D. Blondel, Jean-Loup Guillaume, Renaud Lambiotte, Etienne Lefebvre
https://arxiv.org/abs/0803.0476
Code: https://sourceforge.net/projects/louvain/


Based on Clauset-Newman modularity: https://arxiv.org/pdf/physics/0602124.pdf Modularity and community structure in networks

Link Communities: https://arxiv.org/pdf/0903.3178.pdf


# Download

curl 'http://index.commoncrawl.org/CC-MAIN-2019-47-index?url=www.peoplebank.com.au/job/*&limit=10&output=json' > data.json

cat data.json | jq -r '[.filename, .offset, .length] | @csv' | sed 's/"//g' | awk -F, '{print $1, $2, $2+$3}' > lines


### Writing WARC from cdxt:

```
time cdxt --cc --from 202002 --to 202003 --filter '=status:200' warc --prefix 202002_jobs.launchrecruitement.com.au 'jobs.launchrecruitment.com.au/job/*'
```

10 minutes: 26M

--ia: 2min: 5M (remove = from filter)

```
time cdxt --cc --from 202002 --to 202003 --filter '=status:200' --limit 10 iter --fields url,filename,length,offset 'jobs.launchrecruitment.com.au/job/*' > data.txt
```


# Visual elements

CharGrid

ACL2020 Tutorial 2

# Things
Google AMP

# Coding

Editing Hugo templates

Speed tests

Can, Must, Should (and Ethics)

Google Analytics blockers (privacy badger, ublock, ...)


lrwxrwxrwx 1 eross eross   16 Oct 26 08:24 .#footer-scripts.html -> 'eross@L<numbers>.<numbers>