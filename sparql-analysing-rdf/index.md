---
categories:
- python
- rdf
- commoncrawl
- data
date: '2020-06-25T08:00:00+10:00'
image: /images/rdf_analysis.png
title: Analytics Web Data Commons with SPARQL
---

I am trying to understand how the [JobPosting schema](https://schema.org/JobPosting) is used in [Web Data Commons](http://webdatacommons.org/) structured data extracts from [Common Crawl](https://commoncrawl.org/).
I wrote a lot of ad hoc Python to [get usage statistics on JobPosting](/schema-jobposting).
However SPARQL is a tool that makes it much easier to answer these kinds of questions.

After [reading in the graphs](/streaming-nquad-rdf) individually they can be combined into a [`rdflib.Dataset`](https://rdflib.readthedocs.io/en/stable/apidocs/rdflib.html#rdflib.graph.Dataset) so we can query them all together.
For this analysis I got 13,000 pages from the [2019 Web Data Commons Extract](http://webdatacommons.org/structureddata/2019-12/stats/schema_org_subsets.html), each from a distinct web domain and containing exactly one job posting.

```python
dataset = rdflib.Dataset()
for graph in graphs:
    dataset.add_graph(graph)
```

We can then execute [SPARQL queries in RDFLib](https://rdflib.readthedocs.io/en/stable/intro_to_sparql.html) using `dataset.query`.
The [SPARQL 1.1 Specification](https://www.w3.org/TR/sparql11-query/) is actually pretty easy to read and with a little practice it's a simple language to learn.
Here's a query that gives summary statistics of the most common RDF types in the sample of graphs.


```sparql
SELECT ?type (COUNT(?src) AS ?postings) (SUM(?n) as ?total) {
SELECT ?src ?type (COUNT(?type) AS ?n)
WHERE {
    GRAPH ?src
    {[] a ?type .}
}
GROUP BY ?src ?type
}
GROUP BY ?type
HAVING (COUNT(?src) > 50)
ORDER BY desc(?total)
```

Using Pandas we can make it more meaningful by calculating the proportion of pages with each RDF type, and the average number of times a type occurs in a page graph.

```python
df = pd.DataFrame([[value.toPython() for value in row] for row in results],
                  columns = ['uri', 'n', 'total'])
df.assign(frac=lambda df: df.n/max(df.n),
          avg = lambda df: df.total / df.n)
```
| URI                                 | n     | total | frac     | avg      |
|-------------------------------------|-------|-------|----------|----------|
| http://schema.org/JobPosting        | 13092 | 13092 | 1.000000 | 1.000000 |
| http://schema.org/Place             | 8734  | 9301  | 0.667125 | 1.064919 |
| http://schema.org/Organization      | 7972  | 9184  | 0.608921 | 1.152032 |
| http://schema.org/PostalAddress     | 8065  | 9018  | 0.616025 | 1.118165 |
| http://schema.org/MonetaryAmount    | 2958  | 2970  | 0.225940 | 1.004057 |
| http://schema.org/PropertyValue     | 2875  | 2882  | 0.219600 | 1.002435 |
| http://schema.org/ListItem          | 946   | 2871  | 0.072258 | 3.034884 |
| http://schema.org/QuantitativeValue | 2602  | 2619  | 0.198747 | 1.006533 |
| http://schema.org/ImageObject       | 609   | 1057  | 0.046517 | 1.735632 |
| http://schema.org/BreadcrumbList    | 939   | 988   | 0.071723 | 1.052183 |

The most common object is a JobPosting and we can construct a query to get the most frequently used properties.

```sparql
PREFIX sdo: <http://schema.org/>
SELECT ?rel (COUNT(?src) AS ?postings) (SUM(?n) as ?total) {
SELECT ?rel ?src (COUNT(?src) AS ?n)
WHERE {
    GRAPH ?src
    {[] a sdo:JobPosting; ?pred ?data }
    BIND (replace(str(?pred), 'https?://schema.org/(JobPosting/)?', '') AS ?rel)
}
GROUP BY ?rel ?src
}
GROUP BY ?rel
ORDER BY desc(?postings)
```

| Property                                        | Postings | Total | Fraction of Posts | Average times per post |
|-------------------------------------------------|----------|-------|-------------------|------------------------|
| http://www.w3.org/1999/02/22-rdf-syntax-ns#type | 13092    | 13092 | 1.000000          | 1.000000               |
| title                                           | 11862    | 12052 | 0.906049          | 1.016018               |
| description                                     | 11323    | 11540 | 0.864879          | 1.019165               |
| datePosted                                      | 10420    | 10515 | 0.795906          | 1.009117               |
| jobLocation                                     | 9800     | 10423 | 0.748549          | 1.063571               |
| hiringOrganization                              | 9568     | 9720  | 0.730828          | 1.015886               |
| employmentType                                  | 7702     | 8139  | 0.588298          | 1.056739               |
| validThrough                                    | 4688     | 4691  | 0.358081          | 1.000640               |
| baseSalary                                      | 3657     | 3713  | 0.279331          | 1.015313               |
| industry                                        | 3328     | 4081  | 0.254201          | 1.226262               |
| identifier                                      | 3214     | 3217  | 0.245493          | 1.000933               |
| url                                             | 2744     | 2894  | 0.209594          | 1.054665               |
| workHours                                       | 1352     | 1366  | 0.103269          | 1.010355               |
| experienceRequirements                          | 1235     | 1262  | 0.094332          | 1.021862               |
| occupationalCategory                            | 1152     | 1509  | 0.087993          | 1.309896               |
| educationRequirements                           | 959      | 991   | 0.073251          | 1.033368               |
| salaryCurrency                                  | 904      | 910   | 0.069050          | 1.006637               |
| qualifications                                  | 839      | 902   | 0.064085          | 1.075089               |
| responsibilities                                | 834      | 894   | 0.063703          | 1.071942               |
| image                                           | 790      | 859   | 0.060342          | 1.087342               |
| skills                                          | 726      | 795   | 0.055454          | 1.095041               |


Digging further we could extract the types of the baseSalary; while it's mostly a MonetaryAmount (a complex object) it's also often a string in some language (a literal).

```python
PREFIX sdo: <http://schema.org/>
PREFIX sdo_jp: <http://schema.org/JobPosting/>

SELECT ?type (COUNT(?src) as ?n)
WHERE {
    GRAPH ?src
    {[] a sdo:JobPosting; sdo:baseSalary|sdo_jp:baseSalary ?data .
     OPTIONAL {?data a ?datatype .}
     BIND (coalesce(datatype(?data), ?datatype) as ?type)}
}
GROUP BY ?type
ORDER BY DESC(?n)
LIMIT 20
```

| Type                                                  | Count |
|-------------------------------------------------------|-------|
| http://schema.org/MonetaryAmount                      | 2946  |
| http://www.w3.org/1999/02/22-rdf-syntax-ns#langString | 527   |
| http://www.w3.org/2001/XMLSchema#string               | 135   |
| https://schema.org/MonetaryAmount                     | 72    |
| None                                                  | 16    |
| http://schema.org/PriceSpecification                  | 7     |
| http:/schema.orgMonetaryAmount                        | 6     |

As with JobPosting we can then dig into the most commonly used properties of a MonetaryAmount.

```sparql
PREFIX sdo: <http://schema.org/>
PREFIX sdo_jp: <http://schema.org/JobPosting/>

SELECT ?rel (COUNT(?src) AS ?postings) (SUM(?n) as ?total) {
SELECT ?rel ?src (COUNT(?src) AS ?n)
WHERE {
    GRAPH ?src
    {[] a sdo:JobPosting; sdo:baseSalary|sdo_jp:baseSalary ?salary .
     ?salary ?pred ?data .}
    BIND (replace(str(?pred), 'https?://schema.org/(MonetaryAmount/)?', '') AS ?rel)
}
GROUP BY ?rel ?src
}
GROUP BY ?rel
ORDER BY desc(?postings)
```

| Type                                            | Fraction of all jobs | Fraction of results | Average Frequency |
|-------------------------------------------------|----------------------|---------------------|-------------------|
| http://www.w3.org/1999/02/22-rdf-syntax-ns#type | 0.230734             | 1.000000            | 1.003972          |
| value                                           | 0.215841             | 0.935452            | 1.005308          |
| currency                                        | 0.206675             | 0.895730            | 1.004065          |
| minValue                                        | 0.010922             | 0.047335            | 1.000000          |
| maxValue                                        | 0.010769             | 0.046673            | 1.000000          |
| unitText                                        | 0.002979             | 0.012910            | 1.000000          |

And we can continue into looking at the datatypes of a MonetaryAmount value, the RDF type when it's blank or the Literal type.

```sparql
PREFIX sdo: <http://schema.org/>
PREFIX sdo_jp: <http://schema.org/JobPosting/>
PREFIX sdo_ma: <http://schema.org/MonetaryAmount/>


SELECT ?type (COUNT(?src) as ?n)
WHERE {
    GRAPH ?src
    {[] a sdo:JobPosting; (sdo:baseSalary|sdo_jp:baseSalary)/(sdo:value|sdo_ma:value) ?data .
     OPTIONAL {?data a ?datatype .}
     BIND (coalesce(datatype(?data), ?datatype) as ?type)}
}
GROUP BY ?type
ORDER BY DESC(?n)
LIMIT 20
```

| Type                                                  | n    |
|-------------------------------------------------------|------|
| http://schema.org/QuantitativeValue                   | 2323 |
| http://www.w3.org/1999/02/22-rdf-syntax-ns#langString | 280  |
| http://www.w3.org/2001/XMLSchema#string               | 152  |
| None                                                  | 7    |
| http://schema.org/PropertyValue                       | 4    |
| https://schema.org/QuantitativeValue                  | 2    |


These kinds of techniques could be templated and extended to build a full frequency table.
These are generally fairly consistent with what I found before with a different method and a smaller sample so this increases my confidence in those results.
You can see the [full Jupyter notebook](https://github.com/EdwardJRoss/job-advert-analysis/blob/master/notebooks/JobPosting%20SPARQL%20-%202019%20Web%20Data%20Commons%20Analysis.ipynb) for details.