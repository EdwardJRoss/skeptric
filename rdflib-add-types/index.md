---
categories:
- python
- rdf
date: '2020-06-24T08:00:00+10:00'
image: /images/rdflib_bind.png
title: Adding Types to Rdflib
---

I've been using [RDFLib](https://rdflib.readthedocs.io/) to [parse Job posts extracted from Common Crawl](/streaming-nquad-rdf).
RDF Literals 
It automatically parses XML Schema Datatypes into Python datastructures, but doesn't handle the `<http://schema.org/Date>` datatype that commonly occurs in JSON-LD.
It's easy to add with the [`rdflib.term.bind`](https://rdflib.readthedocs.io/en/stable/apidocs/rdflib.html#rdflib.term.bind) command, but this kind of global binding could lead to problems.

When RDFLib parses a literal it will create a rdflib.term.Literal object and the `value` field will contain the Python type if it can be successfully converted, otherwise it will be `None`.
This object has a `toPython()` method that will return the value unless it is `None`, in which case it will return the object itself.
To see how this works here's some simple code to parse some RDF data and output all the objects: both in raw form, through `toPython` and the `value`.

```python
def parse_objects(data, format='ntriples'):
    G = rdflib.Graph()
    G.parse(data=data, format=format)
    return [(o, o.toPython(), o.value) for o in G.objects()]
```

For a string literal it is represented as the string itself.

```python
parse_objects('_:b <http://example.org/value> "1" . \n')

>> [(rdflib.term.Literal('1'), '1', '1')]
```

For an XML Schema integer it is stored as a Python integer

```python
parse_objects('_:b <http://example.org/value> "1"^^<http://www.w3.org/2001/XMLSchema#integer> . \n')

>> [(rdflib.term.Literal('1', datatype=rdflib.term.URIRef('http://www.w3.org/2001/XMLSchema#integer')),
>>>  1, 1)]
```

Note that if the data doesn't match the type the value remains as None.

```python
parse_objects('_:b <http://example.org/value> "2020-01-01"^^<http://www.w3.org/2001/XMLSchema#integer> . \n')

>> [(rdflib.term.Literal('2020-01-01', datatype=rdflib.term.URIRef('http://www.w3.org/2001/XMLSchema#integer')),
>>>  rdflib.term.Literal('2020-01-01', datatype=rdflib.term.URIRef('http://www.w3.org/2001/XMLSchema#integer')),
>>>  None)]
```

If we have a custom datatype then the value also remains as None.

```python
parse_objects('_:b <http://example.org/value> "2020-01-01"^^<http://schema.org/Date> . \n')

>> [(rdflib.term.Literal('2020-01-01', datatype=rdflib.term.URIRef('http://schema.org/Date')),
>>>  rdflib.term.Literal('2020-01-01', datatype=rdflib.term.URIRef('http://schema.org/Date')),
>>>  None)]
```

We can add a custom datatype with `rdflib.term.bind` that allows converting between Python types and RDF types.
In this case we're only interested in converting from RDF to Python.
The arguments are:

* datatype: The RDF Datatype we want to convert
* pythontype: The corresponding Python datatype
* constructor: How to turn an RDF literal to a Python datatype
* lexicalizer: How to turn a Python datatype to an RDF (not needed here)
* datatype_specific: Whether the binding is specific or general; there are other representations of datetime so set to True

```python
import datetime, dateutil
rdflib.term.bind(datatype=rdflib.URIRef('http://schema.org/Date'),
                 pythontype=datetime.datetime,
                 constructor=dateutil.parser.isoparse,
                 lexicalizer= lambda dt: dt.isoformat(),
                 datatype_specific=True)
```

Then running the exact same code now gives a different result, with the correct type.

```python
parse_objects('_:b <http://example.org/value> "2020-01-01"^^<http://schema.org/Date> . \n')

>> [(rdflib.term.Literal('2020-01-01T00:00:00', datatype=rdflib.term.URIRef('http://schema.org/Date')),
>>>  datetime.datetime(2020, 1, 1, 0, 0),
>>>  datetime.datetime(2020, 1, 1, 0, 0))]
```

As for native types if it parsing would be an error we get the value still being null:

```python
parse_objects('_:b <http://example.org/value> "1"^^<http://schema.org/Date> . \n')

>> [(rdflib.term.Literal('1', datatype=rdflib.term.URIRef('http://schema.org/Date')),
>>>  rdflib.term.Literal('1', datatype=rdflib.term.URIRef('http://schema.org/Date')),
>>>  None)]
```

I really don't like that running the same inputs gives a different output for a parser.
It can be really hard to reason about what is happening, as this could be set deep in some code.
Even worse if another package I import uses a different binding for this RDF I could break it.
Ideally bindings would be passed in some scope (e.g. using an object), rather than global state.
Since not many things use RDF in practice it's not a big issue, but it is not a robust design - if this was used in YAML parsing it could be catastrophic.