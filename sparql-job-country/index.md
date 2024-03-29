---
categories:
- jobs
- commoncrawl
- rdf
date: '2020-06-26T08:00:00+10:00'
image: /images/sparql_australian_jobs.png
title: Extracting Australian Job Postings with SPARQL
---

I am trying to extract Australian Job Postings from [Web Data Commons](http://webdatacommons.org/) which extracts structured data from [Common Crawl](https://commoncrawl.org/).
I have previously written scripts to [read in the graphs](/streaming-nquad-rdf), [explore JobPosting schema](/schema-jobposting) and [analyst the schema using SPARQL](/sparql-analysing-rdf).
Now we can use these to find some Austrlian Job Postings in the data.

For this analysis I used 15,000 pages containing job postings with different domains from [the 2019 Web Data Commons Extract](http://webdatacommons.org/structureddata/2019-12/stats/schema_org_subsets.html).
Here's the final query that extracts 285 domains; the rest of this article will explain what it's doing.

```sparql
PREFIX sdo: <http://schema.org/>
PREFIX sdo_jp: <http://schema.org/JobPosting/>
PREFIX sdo_pl: <http://schema.org/Place/>
PREFIX sdo_pa: <http://schema.org/PostalAddress/>
PREFIX sdo_co: <http://schema.org/Country/>
PREFIX sdo_mv: <http://schema.org/MonetaryValue/>
PREFIX sdos_mv: <https://schema.org/MonetaryValue/>


SELECT distinct ?src
WHERE {
 { 
  GRAPH ?src
  {[] a sdo:JobPosting .}
  BIND (replace(str(?src), 
                'https?://([^?/]+).*',
                '\\1') AS ?domain)
    FILTER (strends(?domain, '.au'))
 }
 UNION
 {
  GRAPH ?src
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
}
```

# Finding Australian Domains

The fact there *is* a URL in the JobPostings extract in Web Data Commons tells you that the URL contains a structured Job Posting.
One heuristic for finding Australian job listings is looking for domains ending in `.au`.

We can get the URL from Common Crawl containing the data by searching for the graph identifier, which we'll call `?src`, filtering to graphs containing a JobPosting.

```sparql
PREFIX sdo: <http://schema.org/>

SELECT ?src
WHERE {
    GRAPH ?src
    {[] a sdo:JobPosting .}
}
LIMIT 10
```

This gives a list of 10 URLs like `https://tire-factory.hiringthing.com/job/17125/warehouse-associate`.

We can extract the domain with a regular expression using the [replace function](https://www.w3.org/TR/sparql11-query/#func-replace), and just get results that end in `.au` using [strends (string-ends)](https://www.w3.org/TR/sparql11-query/#func-strends).

```sparql
PREFIX sdo: <http://schema.org/>

SELECT DISTINCT ?src
WHERE {
    GRAPH ?src
    {[] a sdo:JobPosting .}
    BIND (replace(str(?src), 'https?://([^?/]+).*', '\\1') AS ?domain)
    FILTER (strends(?domain, '.au'))
}
```

This gets 233 URLs containing job postings; but we can do a little better if we use the structured data in the job postings.

# Finding Job Postings located in Australia

The [JobPosting Schema](https://schema.org/JobPosting) contains a jobLocation, which can be a [Place](https://schema.org/Place) which can contain an address, which can be a [PostalAddress](https://schema.org/PostalAddress), which can contain an addressCountry, which can be a [Country](https://schema.org/Country) which can have a name. Phew!

## Extracting Country Name with property paths

We can express this succinctly using SPARQL [property paths](https://www.w3.org/TR/sparql11-query/#propertypaths).

```sparql
PREFIX sdo: <http://schema.org/>

SELECT ?country (COUNT(distinct ?src) as ?count)
WHERE {
    GRAPH ?src
    {[] a sdo:JobPosting; sdo:jobLocation/sdo:address/sdo:addressCountry/sdo:name ?country .}
}
GROUP BY ?country
ORDER BY DESC(?count)
LIMIT 15
```

| Country | count |
|---------|-------|
| US      | 127   |
| CA      | 20    |
| DE      | 20    |
| GB      | 18    |
| IL      | 14    |

Note that we count distinct graph identifiers; because a page can contain multiple job listings (which in turn can contain multiple jobLocations) it may contribute to multiple countries.

## Extracting plain text addressCountry


The addressCountry can also be plain text, and in fact that's much more common.
We can filter out the cases where it's a structured value (and so `?country` is a blank node) using [isLiteral](https://www.w3.org/TR/sparql11-query/#func-isLiteral).

```sparql
PREFIX sdo: <http://schema.org/>

SELECT ?country (COUNT(distinct ?src) AS ?count)
WHERE {
    GRAPH ?src
    {[] a sdo:JobPosting; sdo:jobLocation/sdo:address/sdo:addressCountry ?country .}
    FILTER (isLiteral(?country))
}
GROUP BY ?country
ORDER BY DESC(?count)
LIMIT 10
```

In the results we can see 86 jobs with AU in the countries.

| Country        | Count |
|----------------|-------|
| United States  | 385   |
| JP             | 358   |
| GB             | 345   |
| US             | 320   |
| DE             | 270   |
| NL             | 253   |
| Deutschland    | 179   |
| United Kingdom | 139   |
| FR             | 110   |
| AU             | 86    |

## Matching Country and text at the same time

We *should* be able to combine the two by making `name` optional with the ZeroOrOnePath operator `?`.

```sparql
PREFIX sdo: <http://schema.org/>

SELECT ?country (count(distinct ?src) as ?count)
WHERE {
    GRAPH ?src
    {[] a sdo:JobPosting; sdo:jobLocation/sdo:address/sdo:addressCountry/(sdo:name?) ?country .}
}
GROUP BY ?country
ORDER BY DESC(?total)
LIMIT 15
```

However for some strange reason we end up with some URIs in the results:

| Country                      | Count |
|------------------------------|-------|
| US                           | 447   |
| United States                | 385   |
| GB                           | 363   |
| JP                           | 359   |
| DE                           | 290   |
| NL                           | 257   |
| Deutschland                  | 179   |
| United Kingdom               | 140   |
| FR                           | 116   |
| AU                           | 91    |
| CA                           | 81    |
| India                        | 60    |
|                              | 60    |
| http://schema.org/JobPosting | 56    |
| http://schema.org/Place      | 56    |

Oddly enough this *doesn't* happen if we rewrite it as an alternation:

```
PREFIX sdo: <http://schema.org/>

SELECT ?country (count(?src) as ?total)
WHERE {
    GRAPH ?src
    {[] a sdo:JobPosting; sdo:jobLocation/sdo:address/(sdo:addressCountry|sdo:addressCountry/sdo:name) ?country .}
}
GROUP BY ?country
ORDER BY DESC(?total)
LIMIT 15
```

| Country        | Total |
|----------------|-------|
| US             | 609   |
| JP             | 444   |
| United States  | 395   |
| GB             | 363   |
| DE             | 304   |
| NL             | 258   |
| Deutschland    | 179   |
| United Kingdom | 140   |
| FR             | 116   |
| AU             | 91    |
| CA             | 84    |
| India          | 70    |
| Canada         | 66    |
|                | 60    |
| IN             | 57    |

I would expect these to be the same; but I don't know if my understanding of SPARQL is wrong or it's a bug in rdflib.
When we filter to literal nodes we get the same results, so I'm not going to dwell on it.

## Fully qualified paths

In the microdata extract the properties are specified by fully qualified paths, for example `<http://schema.org/Place/address>` instead of just `<http://schema.org/address>`.
So we need to match these patterns too, which means adding a whole heap more prefixes.

We can check the property it's binding on, but have to be careful to filter out common strings to reduce false positives (e.g. if `?country` is the empty string then this will extract all properties with an empty string).

```sparql
PREFIX sdo: <http://schema.org/>
PREFIX sdo_jp: <http://schema.org/JobPosting/>
PREFIX sdo_pl: <http://schema.org/Place/>
PREFIX sdo_pa: <http://schema.org/PostalAddress/>
PREFIX sdo_co: <http://schema.org/Country/>

SELECT ?property (count(distinct ?src) as ?count)
WHERE {
    GRAPH ?src
    {[] a sdo:JobPosting ;
         sdo_jp:jobLocation/sdo_pl:address/sdo_pa:addressCountry/(sdo_co:name?) ?country .
         [] ?property ?country .
         FILTER (isliteral(?country) &&
                (lcase(str(?country)) not in ('', 'na', 'n/a', 'unavailable', ' ', 'null')))
         }
}
GROUP BY ?property
ORDER BY DESC(?count)
LIMIT 10
```

Having a Country is very rare in microdata, but this looks about right.

| Property                                        | Count |
|-------------------------------------------------|-------|
| http://schema.org/PostalAddress/addressCountry  | 1351  |
| http://schema.org/Country/name                  | 4     |
| http://schema.org/PostalAddress/addressLocality | 3     |
| http://schema.org/PostalAddress/streetAddress   | 1     |
| http://schema.org/PostalAddress/addressRegion   | 1     |


## Combining the patterns

We can combine the two possible schema paths using alternations.

```
PREFIX sdo: <http://schema.org/>
PREFIX sdo_jp: <http://schema.org/JobPosting/>
PREFIX sdo_pl: <http://schema.org/Place/>
PREFIX sdo_pa: <http://schema.org/PostalAddress/>
PREFIX sdo_co: <http://schema.org/Country/>

SELECT ?country (count(distinct ?src) as ?count)
WHERE {
    GRAPH ?src
    {[] a sdo:JobPosting ;
         (sdo:jobLocation|sdo_jp:jobLocation)/
         (sdo:address|sdo_pl:address)/
         (sdo:addressCountry|sdo_pa:addressCountry)/
         ((sdo:name|sdo_co:name)?) ?country .
         FILTER (isliteral(?country))
         }
}
GROUP BY ?country
ORDER BY DESC(?count)
LIMIT 10
```

Unfortunately sometimes the country has a language tag and this means the results are treated differently.

| Country                 | Count |
|-------------------------|-------|
| United States (Lang=EN) | 469   |
| US                      | 463   |
| United States           | 393   |
| GB                      | 365   |
| JP                      | 359   |
| DE                      | 295   |
| RU                      | 283   |
| NL                      | 257   |
| Deutschland             | 186   |
| United Kingdom          | 142   |

We can strip away the language tags by converting it to a plain string with `str`.
Furthermore we can remove any leading/trailing whitespace with a regular expression.

```sparql
PREFIX sdo: <http://schema.org/>
PREFIX sdo_jp: <http://schema.org/JobPosting/>
PREFIX sdo_pl: <http://schema.org/Place/>
PREFIX sdo_pa: <http://schema.org/PostalAddress/>
PREFIX sdo_co: <http://schema.org/Country/>

SELECT ?countryplain (count(distinct ?src) as ?count)
WHERE {
    GRAPH ?src
    {[] a sdo:JobPosting ;
         (sdo:jobLocation|sdo_jp:jobLocation)/
         (sdo:address|sdo_pl:address)/
         (sdo:addressCountry|sdo_pa:addressCountry)/
         ((sdo:name|sdo_co:name)?) ?country .
         FILTER (isliteral(?country))
         BIND (replace(str(?country), '[ \n\t]*(.*)[ \n\t]*', '\\1') as ?countryplain)
         }
}
GROUP BY ?countryplain
HAVING (COUNT(distinct ?src) >= 50)
ORDER BY DESC(?count)
```

There's still some normalisation to do; United States, US and USA are all the same as are DE, Deutschland and Germany.

| Country        | Count |
|----------------|-------|
| United States  | 863   |
| US             | 496   |
| GB             | 381   |
| JP             | 362   |
| DE             | 355   |
| RU             | 287   |
| NL             | 264   |
| Deutschland    | 192   |
| United Kingdom | 175   |
| FR             | 128   |
| AU             | 96    |
| CA             | 88    |
| India          | 65    |
| Canada         | 61    |
|                | 60    |
| IN             | 59    |
| Germany        | 50    |
| USA            | 50    |

We find jobs located in Australia looking for the country being 'AU' or 'Australia' in some case, after trimming whitespace.

```
PREFIX sdo: <http://schema.org/>
PREFIX sdo_jp: <http://schema.org/JobPosting/>
PREFIX sdo_pl: <http://schema.org/Place/>
PREFIX sdo_pa: <http://schema.org/PostalAddress/>
PREFIX sdo_co: <http://schema.org/Country/>

SELECT DISTINCT ?src
WHERE {
    GRAPH ?src
    {[] a sdo:JobPosting ;
         (sdo:jobLocation|sdo_jp:jobLocation)/
         (sdo:address|sdo_pl:address)/
         (sdo:addressCountry|sdo_pa:addressCountry)/
         ((sdo:name|sdo_co:name)?) ?country .
         FILTER (isliteral(?country) &&
                 lcase(replace(str(?country),
                       '[ \n\t]*(.*)[ \n\t]*', '\\1'))
                 in ('au', 'australia'))
         }
}
```

This gets 124 URLs, 40 of which don't end in `.au`.
This includes some New Zealand job sites, some global companies, some talent platforms with company subdomains for Australian companies ([breezy.hr](https://breezy.hr/), [gosnaphot](https://www.snaphop.com/) and [recruitee](https://recruitee.com/en) and [jobsindevenport.com](https://www.jobsindevonport.com/) which is a site dedicated to jobs in the city of Devonport in Tasmania.
The majority of these look like Australian job ads.

Note that this means that around half of the jobs in a `.au` domain don't have Australia as a country.
I'm willing to guess this is because the metadata is incomplete; they probably don't have an `addressCountry` property at all.

Another place to look for a location would be [applicantLocationRequirements](https://schema.org/applicantLocationRequirements) which is used for remote jobs, but isn't used much in practice and so doesn't seem worth investigating.

# Jobs paying Australian Dollars

Australia has it's own unique currency, the Australian Dollar (AUD).
We could try to find Australian jobs by extracting the currency from the job and matching to AUD.

The easiest way is with the `salaryCurrency` field, removing any lanugage tags as before.

```sparql
PREFIX sdo: <http://schema.org/>

SELECT ?curr (COUNT(distinct ?src) as ?count)
WHERE {
    GRAPH ?src
    {[] a sdo:JobPosting ;
        sdo:salaryCurrency ?currency .
    }
    BIND (str(?currency) as ?curr)
}
GROUP BY ?curr
ORDER BY DESC(?count)
LIMIT 10
```

| Currency | Count |
|----------|-------|
| GBP      | 179   |
| EUR      | 93    |
| USD      | 69    |
| €        | 58    |
| AUD      | 41    |
| JPY      | 27    |
|          | 13    |
| 円       | 8     |
| INR      | 7     |
| HKD      | 7     |

Another way the currency can be encoded is as the `currency` in the `baseSalary`:

```sparql
PREFIX sdo: <http://schema.org/>

SELECT ?curr (COUNT(distinct ?src) AS ?count)
WHERE {
    GRAPH ?src
    {[] a sdo:JobPosting ;
        sdo:baseSalary/sdo:currency ?currency .
    }
    BIND (str(?currency) as ?curr)
}
GROUP BY ?curr
ORDER BY DESC(?count)
LIMIT 10
```

| Currency | Count |
|----------|-------|
| GBP      | 314   |
| JPY      | 261   |
| USD      | 234   |
| EUR      | 211   |
|          | 117   |
| INR      | 102   |
| JPN      | 93    |
| €        | 62    |
| AUD      | 54    |
| AFA      | 23    |


## Combining all the currency variants

As before we add the fully qualified schemas to get every possible variation.
We also add `<https://schema.org/MonetaryValue/>` because this occurs a few times in practice.
In fact the `https://schema.org` [should be equivalent](https://schema.org/docs/faq.html#19) so I should check it everywhere doubling the number of variants.
It doesn't occur much in this dataset, so I mostly ignore it here, but it might become a bigger issue in future.

```
PREFIX sdo: <http://schema.org/>
PREFIX sdo_jp: <http://schema.org/JobPosting/>
PREFIX sdo_mv: <http://schema.org/MonetaryValue/>
PREFIX sdos_mv: <https://schema.org/MonetaryValue/>


SELECT ?curr (COUNT(distinct ?src) as ?count)
WHERE {
    GRAPH ?src
    {[] a sdo:JobPosting ;
        ((sdo:salaryCurrency|sdo_jp:salaryCurrency)|
         (sdo:baseSalary|sdo_jp:baseSalary)/
          (sdo:currency|sdo_mv:currency|sdos_mv:currency)) ?currency .
    }
    BIND (replace(str(?currency), '[ \n\t]+', '') as ?curr)
    FILTER (!(lcase(?curr) in ('', 'null', 'na', 'n/a', 'unavailable')))
}
GROUP BY ?curr
ORDER BY DESC(?count)
LIMIT 20
```

The resulting data is pretty good; we could further normalise € as EUR and £ as GBP, but the currencies otherwise look like [ISO 4217 currency codes](https://en.wikipedia.org/wiki/ISO_4217).

| Currency | Count |
|----------|-------|
| GBP      | 392   |
| USD      | 302   |
| EUR      | 295   |
| JPY      | 266   |
| AUD      | 114   |
| INR      | 108   |
| JPN      | 93    |
| €        | 68    |
| CZK      | 57    |
| RUB      | 50    |
| RUR      | 49    |
| AFA      | 23    |
| CAD      | 19    |
| VND      | 18    |
| HKD      | 14    |
| £        | 14    |
| BRL      | 12    |
| SEK      | 11    |
| PKR      | 10    |
| THB      | 10    |

Finally we can filter down to the 114 Job ads offering salary in AUD:

```sparql
PREFIX sdo: <http://schema.org/>
PREFIX sdo_jp: <http://schema.org/JobPosting/>
PREFIX sdo_mv: <http://schema.org/MonetaryValue/>
PREFIX sdos_mv: <https://schema.org/MonetaryValue/>

SELECT distinct ?src
WHERE {
    GRAPH ?src
    {[] a sdo:JobPosting ;
        ((sdo:salaryCurrency|sdo_jp:salaryCurrency)|
         (sdo:baseSalary|sdo_jp:baseSalary)/(sdo:currency|sdo_mv:currency|sdos_mv:currency)) ?currency .
    }
    BIND (replace(str(?currency), '[ \n\t]+', '') as ?curr)
    FILTER (lcase(?curr) = 'aud')
}
```

This gives 114 jobs, of which 18 don't have a `.au` domain and 12 of those don't have Australia as a country.
Most of these jobs are valid Australian jobs, but for some reason there are a few New Zealand jobs (which *should* be in NZD).

# Combining all the results

The query at the start of the article is just the [`UNION`](https://www.w3.org/TR/sparql11-query/#alternatives) of the three variants: `.au` domain, Australia as a country or AUD as the currency.

One catch is that we need to specify the `.au` in a separate GRAPH query because it filters the domain, which we *don't* want to do for country or salary.

```sparql
 { 
  GRAPH ?src
  {[] a sdo:JobPosting .}
  BIND (replace(str(?src), 
                'https?://([^?/]+).*',
                '\\1') AS ?domain)
    FILTER (strends(?domain, '.au'))
 }
```

But the Country and Salary queries can be done in the same GRAPH search

```sparql
 {
  GRAPH ?src
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
``` 

You can see the [very similar Jupyter notebook](https://github.com/EdwardJRoss/job-advert-analysis/blob/master/notebooks/Extracting%20Australian%20Job%20Ads%20from%20Web%20Data%20Commons%20with%20SPARQL.ipynb) for all the underlying code and analysis.

Now that we have a way of identifying metadata relating to Australian jobs we can start to build them into a pipeline to extract and analyse the data.