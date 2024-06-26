---
categories:
- python
- data
date: '2020-06-22T08:00:00+10:00'
image: /images/Rdf_graph_for_Eric_Miller.png
title: Converting RDF to Dictionary
---

The [Web Data Commons](http://webdatacommons.org/) has a vast repository of structured [RDF Data](https://en.wikipedia.org/wiki/Resource_Description_Framework) about local businesses, hostels, job postings, products and many other things from the internet.
Unfortunately it's not in a format that's easy to do analysis on.
We can [stream the nquad format to get RDFlib Graphs](/streaming-nquad-rdf), but we still need to convert the data into a form we can do analysis on.
We'll do this by turning the relations into dictionaries of properties to the list of objects they contain.

This turns something like this:

```
_:genid2d8020c9b7d2294a778072a41d6d59640a2db0 <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <http://schema.org/JobPosting> <http://jobs.anixter.com/jobs/inventory-management/glenview-il-60026-/category-manager-prof-audio-visual-solutions/153414552962719?lang=en_us> .
_:genid2d8020c9b7d2294a778072a41d6d59640a2db0 <http://schema.org/identifier> _:genid2d8020c9b7d2294a778072a41d6d59640a2db2 <http://jobs.anixter.com/jobs/inventory-management/glenview-il-60026-/category-manager-prof-audio-visual-solutions/153414552962719?lang=en_us> .
_:genid2d8020c9b7d2294a778072a41d6d59640a2db0 <http://schema.org/title> "Category Manager - Prof. Audio Visual Solutions" <http://jobs.anixter.com/jobs/inventory-management/glenview-il-60026-/category-manager-prof-audio-visual-solutions/153414552962719?lang=en_us> .
...
```

Into something like this (which is still not ideal but much closer to usable):

```
{
 'http://schema.org/employmentType': ['FULL_TIME'],
 'http://schema.org/datePosted': [datetime.datetime(2019, 8, 1, 17, 48, 55)],
 'http://schema.org/description': [...]
 'http://schema.org/jobLocation': [{
   'http://www.w3.org/1999/02/22-rdf-syntax-ns#type': ['http://schema.org/Place'],
   'http://schema.org/address': [
     'http://schema.org/addressCountry': ['United States'],
     'http://www.w3.org/1999/02/22-rdf-syntax-ns#type': ['http://schema.org/PostalAddress'],
     'http://schema.org/addressLocality': ['Glenview'],
     'http://schema.org/postalCode': ['60026'],
     'http://schema.org/addressRegion': ['IL']}]}],
 'http://www.w3.org/1999/02/22-rdf-syntax-ns#type': ['http://schema.org/JobPosting'],
 'http://schema.org/validThrough': [datetime.datetime(2019, 11, 11, 0, 0)],
 'http://schema.org/title': ['Category Manager - Prof. Audio Visual Solutions'],
 'http://schema.org/identifier': [
   'http://schema.org/name': ['Anixter International'],
   'http://www.w3.org/1999/02/22-rdf-syntax-ns#type': ['http://schema.org/PropertyValue'],
   'http://schema.org/value': ['inventory-management/glenview-il-60026-/category-manager-prof-audio-visual-solutions/153414552962719']}],
 'http://schema.org/hiringOrganization': [
   'http://schema.org/name': ['Anixter International'],
   'http://www.w3.org/1999/02/22-rdf-syntax-ns#type': ['http://schema.org/Organization']}]}
```

The idea is pretty simple.
We're going to take each job posting object, and map each of its properties to the list of objects with which it's associated.
If one of those objects is a blank node we will recursively transform it into a dictionary of properties to the list of objects with which it's associated.
The resulting code is quite simple:


```python
class CycleError(Exception): pass
    
def graph_to_dict(graph, root, seen=frozenset()):
    result = {}
    for predicate, obj in graph.predicate_objects(root):
        predicate = predicate.toPython()
        if obj in seen:
            raise CycleError(
                f"Cyclic reference to {obj} in {graph.identifier}")
        elif type(obj) == rdflib.term.BNode:
            obj = graph_to_dict(graph, obj, seen.union([obj]))
        else:
            obj = obj.toPython()
        result[predicate] = result.get(predicate, []) + [obj]
    return result
```

We also need a way of extracting the nodes we want at the root of the document.
One simple way to do this is to extract all nodes of a given schema.org type:

```python
from rdflib.namespace import Namespace
SDO_NAMESPACES = [Namespace('http://schema.org/'), Namespace('https://schema.org/')]
def get_blanks_of_sdo_type(graph, sdo_type):
    for namespace in SDO_NAMESPACES:
        rdf_type = namespace[sdo_type]
        for subject in graph.subjects(rdflib.namespace.RDF.type, rdf_type):
            if type(subject) == rdflib.term.BNode:
                yield subject
```

We could then use this, for example, to get all job postings:

```python
def get_job_postings(graph):
    return get_blanks_of_sdo_type(graph, 'JobPosting')
```

These can be stitched together to take in an nquads file and a type and output all graphs rooted of that type.


```python
def extract_nquads_of_type(lines, sdo_type):
    for graph in parse_nquads(lines):
        for node in get_blanks_of_sdo_type(graph, sdo_type):
            yield graph_to_dict(graph, node)
```

# RDF Graph to dictionary

An RDF consists of a set of triples (subject, predicate, object).
Subject is typically a blank node, which you can think of like a variable.
A predicate is typically a URI which describes the kind of relation, like `<http://schema.org/Organization/name>`.
An object can either be another blank node, a URI or a Literal (which can be a string in some language, or a value like an integer or a date).

The idea is given a blank subject node we can represent it by all of its relations to other predicates and objects.
Because a predicate can appear multiple times (e.g. a job could have two locations that you're working at) in general we can represent it as mapping from a predicate to a list of objects.
For a URI or Literal we can represent them directly as Python objects, and for a blank node we can transform that into a dictionary mapping predicates to lists of objects.


The code to do this is fairly simple.
Given a blank node `root` we can get all the corresponding predicates and objects with `graph.predicate_objects(root)`.
It the object is a blank node then we run `graph_to_dict` on that node to expand it into a dictionary, otherwise we convert it to a Python object with `toPython`.
Finnally append each object to the list of predicates.

```python
def graph_to_dict(graph, root)):
    result = {}
    for predicate, obj in graph.predicate_objects(root):
        if type(obj) == rdflib.term.BNode:
            obj = graph_to_dict(graph, obj, seen.union([obj]))
        else:
            obj = obj.toPython()
        predicate = predicate.toPython()
        result[predicate] = result.get(predicate, []) + [obj]
    return result
```

There's one thing I worry about: will this terminate?
What if there's some nasty data with a cycle in it.
Then it will recurse until we exceed the maximum stack depth.
Here's some example data.

```python
from rdflib import URIRef, BNode, Literal

isa = rdflib.term.URIRef("http://example.org/is_a")
nota = rdflib.term.URIRef("http://example.org/not_a")

thing = BNode()  # a GUID is generated
other_thing = BNode()

tautology_graph = rdflib.Graph()

tautology_graph.add((thing, isa, thing))

cycle_graph = rdflib.Graph()
cycle_graph.add((thing, nota, other_thing))
cycle_graph.add((other_thing, nota, thing))
```

The easiest way to avoid this is to track the blank nodes on the path we've seen; if we see one of those nodes again then we've hit a cycle and should raise an error.

This gives the original code.
Note that undefined objects will be represented as an empty dictionary.


```python
class CycleError(Exception): pass
    
def graph_to_dict(graph, root, seen=frozenset()):
    result = {}
    for predicate, obj in graph.predicate_objects(root):
        predicate = predicate.toPython()
        if obj in seen:
            raise CycleError(
                f"Cyclic reference to {obj} in {graph.identifier}")
        elif type(obj) == rdflib.term.BNode:
            obj = graph_to_dict(graph, obj, seen.union([obj]))
        else:
            obj = obj.toPython()
        result[predicate] = result.get(predicate, []) + [obj]
    return result
```

# Finding root nodes

Now that we can convert a blank node to a dictionary, we need a way to find the relevant blank nodes.
One strategy is just to get all the blank node subjects:

```python
def get_blank_subjects(graph):
    """Returns all blank subjects in graph"""
    return frozenset(s for s in graph.subjects() if type(s) == rdflib.term.BNode)
```

But some subjects are just used as objects of other subjects (e.g. an address in a job listing); so we could just use the ones that are not an object.

```python
def get_blank_objects(graph):
    """Returns all blank objects in graph"""
    return frozenset(o for o in graph.objects() if type(o) == rdflib.term.BNode)


def get_root_blanks(graph):
    """Returns all blank nodes that are not objects of any triple"""
    return get_blank_subjects(graph) - get_blank_objects(graph)
```

A different strategy is to get all the items of a certain RDF type.
Many objects have some RDF type which is defined in schema.org:

```python
from rdflib.namespaces import RDF, SDO
def get_blanks_of_sdo_type(graph, sdo_type):
    rdf_type = SDO[sdo_type]
    for subject in graph.subjects(RDF.type, rdf_type):
        if isinstance(subject, rdflib.term.BNode):
            yield subject
```

There's one problem with this: SDO refers to the namespace `https://schema.org/`, whereas in practice most of the nodes use `http://schema.org`.
This means we lose most of our nodes using this approach!
According to the [schema.org FAQ](https://schema.org/docs/faq.html#19) either is fine, and I've [raised an issue in RDFLib](https://github.com/RDFLib/rdflib/issues/1120) to discuss it.
In the meantime we have to manually try both:

```python
from rdflib.namespace import Namespace
SDO_NAMESPACES = [Namespace('http://schema.org/'), Namespace('https://schema.org/')]
def get_blanks_of_sdo_type(graph, sdo_type):
    for namespace in SDO_NAMESPACES:
        rdf_type = namespace[sdo_type]
        for subject in graph.subjects(rdflib.namespace.RDF.type, rdf_type):
            if type(subject) == rdflib.term.BNode:
                yield subject
```

# Stitching it all together

We can stick these together with our [nquad streaming solution](/streaming-nquad-rdf) to get dictionaries of lists of a given type.

```python
def extract_nquads_of_type(lines, sdo_type):
    for graph in parse_nquads(lines):
        for node in get_blanks_of_sdo_type(graph, sdo_type):
            yield graph_to_dict(graph, node)
            
with gzip.open('nquads.gz', 'rt') as f:
  # Extract all job postings
  for d in extract_nquads_of_type(f, 'JobPosting'):
    ...
```

There's still a few things that need to be done to make this usable.
There are many properties that almost always occur only once (although this doesn't seem to be in the specification at all), and so it would be much nicer to represent it as a single object and not a list.
Also the property names are typically very long, being URIs and it would be useful to shorten them.
But now we have the data in a form where we can analyse them and build better transformations for each schema.