---
categories:
- data
- python
date: '2020-06-21T08:00:34+10:00'
image: /images/nquad_spec.png
title: Streaming n-quads as RDF
---

The [Web Data Commons](http://webdatacommons.org/) extracts structured [RDF Data](https://en.wikipedia.org/wiki/Resource_Description_Framework) from about one monthly [Common Crawl](https://commoncrawl.org) per year.
These contain a vast amount of structured information about local businesses, hostels, job postings, products and many other things from the internet.
Python's [RDFLib](https://rdflib.readthedocs.io/en/stable/) can read the n-quad format the data is stored in, but by default requires reading all of the millions to billions of relations into memory.
However it's possible to process this data in a streaming fashion allowing it to be processed much faster.

To get an idea of the kind of data we're talking about there are over 650 thousand pages from over 8,000 domains containing a Job Posting like this:

```
{'type': 'JobPosting',
 'title': 'Category Manager - Prof. Audio Visual Solutions',
 'jobLocation': {'address': {
     'addressRegion': 'IL',
     'addressLocality': 'Glenview',
     'addressCountry', 'United States',
     'postalCode': '60026',}}
 'hiringOrganization': {'name': 'Anixter International'},
 'employmentType': 'FULL_TIME',
 'datePosted': '2019-08-01 17:48:55',
 'validThrough': '2019-11-11',
 'description': ...,
}
```

However it's stored in large files with millions of lines like this:

```
_:genid2d8020c9b7d2294a778072a41d6d59640a2db0 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/JobPosting> <http://jobs.anixter.com/jobs/inventory-management/glenview-il-60026-/category-manager-prof-audio-visual-solutions/153414552962719?lang=en_us> .
_:genid2d8020c9b7d2294a778072a41d6d59640a2db0 <http://schema.org/identifier> _:genid2d8020c9b7d2294a778072a41d6d59640a2db2 <http://jobs.anixter.com/jobs/inventory-management/glenview-il-60026-/category-manager-prof-audio-visual-solutions/153414552962719?lang=en_us> .
_:genid2d8020c9b7d2294a778072a41d6d59640a2db0 <http://schema.org/title> "Category Manager - Prof. Audio Visual Solutions" <http://jobs.anixter.com/jobs/inventory-management/glenview-il-60026-/category-manager-prof-audio-visual-solutions/153414552962719?lang=en_us> .
```

We can extract these using RDFlib with [itertool's groupby](https://docs.python.org/3.7/library/itertools.html#itertools.groupby) to extract the data from each crawled webpage into a separate RDF Graph for further transformation.

```python
from itertools import groupby
import rdflib
def parse_nquads(lines):
    for group, quad_lines in groupby(lines, get_quad_label):
        graph = rdflib.Graph(identifier=group)
        graph.parse(data=''.join(quad_lines), format='nquads')
        yield graph
```

We can extract the crawled URL from the quad with a simple piece of logic:

```python
import re
RDF_QUAD_LABEL_RE = re.compile("[ \t]+<([^ \t]*)>[ \t].\n$")
def get_quad_label(s):
    return RDF_QUAD_LABEL_RE.search(line).group(1)
```

Then we can process the data one graph at a time

```python
import gzip
with gzip.open('afile.nquads.gz', 'rt') as f:
  for graph in parse_nquads(f):
    ...
```


# What's an n-quad?

The data is stored in RDF n-quads which is specified in a [W3C recommendation](https://www.w3.org/TR/n-quads/).
Each line represents a relation of the form: "subject predicate object graphlabel ."

For example the line below:

```
_:genid2d8020c9b7d2294a778072a41d6d59640a2db0 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/JobPosting> <http://jobs.anixter.com/jobs/inventory-management/glenview-il-60026-/category-manager-prof-audio-visual-solutions/153414552962719?lang=en_us> .
```

Has:

* Object `_:genid2d8020c9b7d2294a778072a41d6d59640a2db0` (a blank Node)
* Predicate: `<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>`
* Subject: `<http://schema.org/JobPosting>` 
* Graph Label: `<http://jobs.anixter.com/jobs/inventory-management/glenview-il-60026-/category-manager-prof-audio-visual-solutions/153414552962719?lang=en_us>`

So this is saying that this object has RDF Type Job Posting, and is from the URL in the graph label.

Similarly the line:

```
_:genid2d8020c9b7d2294a778072a41d6d59640a2db0 <http://schema.org/title> "Category Manager - Prof. Audio Visual Solutions" <http://jobs.anixter.com/jobs/inventory-management/glenview-il-60026-/category-manager-prof-audio-visual-solutions/153414552962719?lang=en_us> .
```

Says that the same object has a title "Category Manager - Prof. Audio Visual Solutions".

We want a way to collect all of these objects up together for each source URL (graph label).

While it's not too hard to parse directly from the specification, there are some subtleties (like types of literals) and it's generally better to use a library.
Also being able to relate the nodes some logic, and while we could use [kanren in Python](https://github.com/logpy/logpy), RDFLib is designed specially for the job.

# Python RDFLib

The easiest way to read this is all at once with a [`ConjunctiveGraph`](https://rdflib.readthedocs.io/en/stable/apidocs/rdflib.html?highlight=conjunctivegraph#rdflib.ConjunctiveGraph) like in [this article by Rebecca Bilbro](https://rebeccabilbro.github.io/rdf-basics/).

```python
from rdflib import ConjunctiveGraph

graph = ConjunctiveGraph()
with open(file, 'rt') as f:
    graph.parse(data, format="nquads")
```

Unfortunately this means we have to load all the data into memory at once and process all the graphs from different files together, only to split them apart after.
With gigabytes of compressed n-quads this is not a good solution, and there don't seem to be any [answers on StackOverflow](https://stackoverflow.com/questions/56007110/split-all-the-different-graphs-included-in-a-n-quads-file).

# Extracting a label from the n-quad

By the specification the label, if it is there, should be a URI (e.g. `<http://example.org/page>`) or a blank node.
I'm going to assume that it's always there and always a URI; this is true for Web Data Commons where the label is the URL of the page the data was extracted from.
We can then use the [specification](https://www.w3.org/TR/n-quads/) to construct a simple regular expression to extract it.

```python
import re
RDF_QUAD_LABEL_RE = re.compile("[ \t]+<([^ \t]*)>[ \t].\n$")
def get_quad_label(s):
    return RDF_QUAD_LABEL_RE.search(line).group(1)
```

Ideally we would use RDFLib nquad parser directly to do it, but it mainly uses stateful objects [internally](https://github.com/RDFLib/rdflib/blob/master/rdflib/plugins/parsers/nquads.py) and hides the quad structure so we end up having to do some contortions to use it.
It uses a similar but [more conservative expression](https://github.com/RDFLib/rdflib/blob/master/rdflib/plugins/parsers/ntriples.py#L27) for a URI.

Now we can identify the labels we can group together the lines that have the same label, on the assumption that they are sequential (they seem to be in the Web Commons data).
We could always ensure this by sorting the *reversed* lines (e.g. in bash `rev | sort | rev`).

```python
f = gzip.open('afile.nquad.gz', 'rt')
label_groups = groupby(f, get_quad_label)
group, quad_lines = next(label_groups)
```

# Reading into RDFLib Graphs

By default an RDFlib Graph ignores anything that doesn't match the identifier in the constructor, so we need to set that to the pages URL.

```python
graph = rdflib.Graph(identifier=group)
```

Then we can parse it by passing the lines as a string to `data` (which gets converted to BytesIO under the hood):

```python
graph.parse(data=''.join(quad_lines), format='nquads')
```

Now we have a valid RDFGraph in `graph`, we can [navigate it](https://rdflib.readthedocs.io/en/stable/intro_to_graphs.html) and generally transform and save it one graph at a time.
It's a pity that we have to do this legwork and it's not easy with `rdflib` directly.