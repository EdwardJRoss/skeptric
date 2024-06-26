---
categories:
- data
- commoncrawl
date: '2020-06-28T08:00:00+10:00'
image: /images/nquad_grep.png
title: Processing RDF nquads with grep
---

I am trying to extract Australian Job Postings from [Web Data Commons](http://webdatacommons.org/) which extracts structured data from [Common Crawl](https://commoncrawl.org/).
I previously came up with a [SPARQL query](/sparql-job-country) to extract the Australian jobs from the domain, country and currency.
Unfortunately it's quite slow, but we can speed it up dramatically by replacing it with a similar script in grep.

With a short grep script we can get twenty thousand Australian Job Postings with metadata from 16 million lines of compressed nquad in 30 seconds on my laptop.
This can be run against any [Web Data Commons Extract of Job Postings](http://webdatacommons.org/structureddata/2019-12/stats/schema_org_subsets.html).

```sh
zgrep -aE \
'(<https?://schema.org/([^ >]+/)?(addressCountry|name|salaryCurrency|currency)> "(Australia|AU|AUS|AUD)")|( <https?://[^ >/]+\.au/[^ >]*> \.$)' \
 *.gz | \
 grep -Eo '<https?://[^ >]+> .$' |
 uniq | \
 sed -E 's/<([^ >]+)> .$/\1/' | \
  sort -u > \
 au_2019_urls.txt
```

The top 10 domains (of 450) from this extract look reasonable; recruitment agencies operating in Australia.

| # URLs | Domain                         |
|--------|--------------------------------|
| 1631   | www.davidsonwp.com             |
| 952    | www.cgcrecruitment.com         |
| 809    | www.peoplebank.com.au          |
| 634    | www.people2people.com.au       |
| 610    | www.designandbuild.com.au      |
| 590    | www.perigongroup.com.au        |
| 554    | www.ambition.com.au            |
| 532    | www.medicalrecruitment.com.au  |
| 528    | www.talentinternational.com.au |
| 456    | www.accountability.com.au      |


# Using SPARQL in rdflib

To [stream the nquads into rdflib](/streaming-nquad-rdf) one graph at a time I needed to use a regular expression to get the URL.
This is safe to do because in Web Data Commons the last quad is always the URI that the data was obtained from, and URIs can not contain spaces or tabs.

```python
import re
RDF_QUAD_LABEL_RE = re.compile("[ \t]+<([^ \t]*)>[ \t]+.\n$")
def get_quad_label(s):
    return RDF_QUAD_LABEL_RE.search(line).group(1)
```

If we want to check that the domain is `.au` then we can parse it with urllib.

```python
import urllib
def get_domain(url):
    return urllib.parse.urlparse(url).netloc
```

We can then filter out the Australian domains in about 4 minutes, giving 17,000 distinct URLs.
Because this extract only contains URLs with a JobPosting these will all have structured job ads.

```python
from collections import groupby
import gzip
from tqdm.notebook import tqdm
f = iter(tqdm(gzip.open(JOBS_JSON_2019, 'rt'), total=16_925_915))
au_urls = []
for url, _ in  groupby(f, get_quad_label):
    if get_domain(url).endswith('au'):
        au_urls.append(url)
```

We can extend this with a SPARQL query to search for [Australian country or currency](/sparql-job-country).

```sparql
PREFIX sdo: <http://schema.org/>
PREFIX sdo_jp: <http://schema.org/JobPosting/>
PREFIX sdo_pl: <http://schema.org/Place/>
PREFIX sdo_pa: <http://schema.org/PostalAddress/>
PREFIX sdo_co: <http://schema.org/Country/>
PREFIX sdo_mv: <http://schema.org/MonetaryValue/>
PREFIX sdos_mv: <https://schema.org/MonetaryValue/>


ASK WHERE {
  {
    {[] a sdo:JobPosting ;
         (sdo:jobLocation|sdo_jp:jobLocation)/
         (sdo:address|sdo_pl:address)/
         (sdo:addressCountry|sdo_pa:addressCountry)/
         ((sdo:name|sdo_co:name)?) ?country .
         FILTER (isliteral(?country) && 
                 lcase(replace(str(?country),
                               '[ \n\t]*(.*)[ \n\t]*',
                               '\\1')) in ('au', 'australia'))
    }
    UNION
    {[] a sdo:JobPosting ;
        ((sdo:salaryCurrency|sdo_jp:salaryCurrency)|
         (sdo:baseSalary|sdo_jp:baseSalary)/
         (sdo:currency|sdo_mv:currency|sdos_mv:currency)) ?currency .
    BIND (replace(str(?currency), '[ \n\t]+', '') as ?curr)
    FILTER (lcase(?curr) = 'aud')}
  }
 }
```

Then we can build a filter with this on the graph, and add a function to get the domain for the graph to search for `.au` domains.

```python
def query_au(g):
    result = list(g.query(aus_sparql))[0]
    return result

def graph_domain(g):
  url = graph.identifier.toPython()
  return get_domain(url)
```

We can then apply this filter but it takes 40 minutes; most of this time is parsing the graph with rdflib.

```python
f = iter(tqdm(gzip.open(JOBS_JSON_2019, 'rt'), total=16_925_915))
au_urls = []
for graph in parse_nquads(f):
    if graph_domain(graph).endswith('au') or query_au(graph):
        au_urls.append(graph_url(graph))
```

# Translating to shell

The first script searching for `.au` domains is essentially a regular expression, and can be translated directly into grep and sed.
This takes 30s, so is about 8 times faster than Python.

```sh
zgrep -aEo ' <https?://[^/ >]+\.au/[^ >]*> .$' \
      2019-12_json_JobPosting.gz | \
  uniq | \
  sed -E 's/<([^ >]+)> .$/\1/' | \
  sort -u > \
 au_urls.txt
```

For finding the country or currency we make a simplifying assumption; that if a name/country/currency is "Australia", "AU", "AUS" or "AUD" then it is a job ad located in Australia.
This is a pretty safe assumption; it's unlikely that e.g. a company name would by "Australia" or "AUS".
At worst we can filter out false negatives later with more processing.
We can then extract the URL with sed as before, and it still takes 30s (rather than 40 minutes using rdflib).

```sh
zgrep -E '<https?://schema.org/([^ >]+/)?(addressCountry|name|salaryCurrency|currency)> "(Australia|AU|AUS|AUD)"'
 2019-12_json_JobPosting.gz | \
 grep -Eo '<https?://[^ >]+> .$' |
 uniq | \
 sed -E 's/<([^ >]+)> .$/\1/' | \
  sort -u > \
 au_extra_urls.txt
```

Finally we can combine the two expressions to get the process at the start of the article.
We can then analyse the output with some [shell processing](/shell-etl) to get the top domains at the start of the article.

```sh
sed -E 's|https?://([^ /]+)/.*|\1|'\
    au_urls.txt |\
    sort |\
    uniq -c |\
    sort -nr |\
    head
```