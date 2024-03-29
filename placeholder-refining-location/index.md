---
categories:
- data
date: '2020-08-23T21:31:52+10:00'
image: /images/placeholder_refine.png
title: Refining Location with Placeholder
---

Placeholder is a great library for [Coarse Geocoding](/coarse-geocoding), and I'm using it for finding [locations in Australia](/placeholder-australia).
In my application I want to get the location to a similar level of granularity; however the input may be for a higher level of granularity.
Placeholder doesn't directly provide a method to do this, but you can use their SQLite database to do it.

For example to find the largest locality for East Gippsland, with Who's On First id [102049039](https://spelunker.whosonfirst.org/id/102049039/), you can use the SQL.

```sql
SELECT
  docs.id,
  json_extract(json, '$.name') AS name,
  json_extract(json, '$.population') AS pop,
  json_extract(json, '$.geom.area') AS area
FROM lineage
JOIN docs ON docs.id = lineage.id
WHERE pid = 102049039
AND json_extract(json, '$.placetype') = 'locality'
ORDER BY pop DESC, area DESC
LIMIT 10;
```

# The Placeholder Database

[Placeholder](https://github.com/pelias/placeholder) is an open source library for coarse geocoding.
It is based on an extract of larger entities from Who's on First (perhaps locality and above), and stores them in a SQLite database.

You can download the 1.8GB SQLite database using `curl https://data.geocode.earth/placeholder/store.sqlite3.gz`.
Once the data is downloaded and extracted with `gunzip` you run `sqlite3` in a terminal and open the database with `.open store.sqlite3`.

The contents looks like this:

```
sqlite> .tables
docs              fulltext_content  lineage           rtree_parent
fulltext          fulltext_data     rtree             rtree_rowid
fulltext_config   fulltext_idx      rtree_node        tokens

```
## Docs

The Docs contain all the geojson data for each place; including the id, names, placetype, lineage, geometry, and population.
This is what Placeholder returns from a Query.


```
sqlite> select * from docs limit 1;
id|json
1|{"id":1,"name":"Null Island","placetype":"country","rank":{"min":19,"max":20},"lineage":[{"country_id":1}],"geom":{"bbox":"-0.0005,-0.000282,0.000379,0.000309","lat":0.000003,"lon":0.00001},"names":{"eng":["Null Island"],"epo":["Nulinsulo"],"fra":["Null Island"],"heb":["נאל איילנד"],"hun":["Nulla Sziget"],"ind":["Null Island"],"ita":["Null island"],"jbo":[".nyldaplu."],"jpn":["ヌル島"],"lzh":["虛無島"],"mkd":["Нулти Остров"],"msa":["Pulau Nol"],"rus":["остров Ноль"],"spa":["Null Island"],"ukr":["Острів Нуль"],"vie":["đảo Rỗng"],"zho":["空虛島"]}}
```

Note that SQLite has [methods for extracting from JSON](https://www.sqlite.org/json1.html).
In particular we could extract the main attributes:

```
SELECT
  id,
  json_extract(json, '$.name') AS name,
  json_extract(json, '$.placetype') AS placetype,
  json_extract(json, '$.population') AS pop,
  json_extract(json, '$.geom.area') AS area
FROM docs
LIMIT 5;
```


| id       | name                         | placetype  | pop     | area     |
|----------|------------------------------|------------|---------|----------|
| 1        | Null Island                  | country    |         |          |
|          |                              |            |         |          |
| 85632161 | Macau S.A.R.                 | country    | 449198  | 0.002313 |
| 85632163 | Guam                         | dependency | 178430  | 0.046566 |
| 85632167 | Bahrain                      | country    | 1332171 | 0.070331 |
| 85632169 | United States Virgin Islands | dependency | 109825  | 0.031723 |


## Tokens and Fulltext

Tokens are all the different names for a location in different languages.
This is used in searching by place name.

```
sqlite> select * from tokens limit 5;
id|lang|tag|token
1|eng|preferred|null island
1|und|abbr|xn
1|epo|preferred|nulinsulo
1|heb|preferred|נאל איילנד
1|hun|preferred|nulla sziget
```

More specifically the `fulltext` table is a [full text search](https://www.sqlite.org/fts5.html) on the term table, with words separated by an underscore.
It is row aligned with the term table so you can use fulltext search to get relevant ids.

```sql
SELECT id
FROM tokens as t1
  JOIN fulltext AS f1 ON f1.rowid = t1.rowid
WHERE f1.fulltext MATCH 'nulla_sziget'
```

## Lineage

Who's on First has a notion of lineage; [Rockhampton](https://spelunker.whosonfirst.org/id/102048825/) is a county in the region of Queensland in the country of Australia in the continent of Oceania.
This is recorded in the lineage table, for each `id` each ancestor is in a row with a `pid`.
For example searching for the ancestors of Rockhampton gives:

```sql
SELECT
  lineage.pid,
  token
FROM lineage
JOIN tokens ON lineage.pid = tokens.id
WHERE lineage.id = 102048825
AND lang = 'eng'
AND tag = 'preferred';
```

| pid       | token      |
|-----------|------------|
| 85632793  | australia  |
| 85681463  | queensland |
| 102191583 | oceania    |

## Rtree

Sometimes the lineage alone doesn't capture the search and so Placeholder also stores the rectangular bounding boxes in an [R-tree](https://en.wikipedia.org/wiki/R-tree) for efficient searching.
For example to search for locations within 0.1 degrees of Rockhampton you could query it like this:

```
SELECT t2.id AS id, t2.token as token
FROM rtree AS r1, rtree AS r2
  JOIN tokens AS t2 ON t2.id = r2.id
WHERE r1.id = 102048825
AND lang = 'eng' AND tag = 'preferred'
-- https://silentmatt.com/rectangle-intersection/
AND (
  r1.maxZ > r2.minZ AND
  r1.minX - 0.1 < r2.maxX AND
  r1.maxX + 0.1 > r2.minX AND
  r1.minY - 0.1 < r2.maxY AND
  r1.maxY + 0.1 > r2.minY
)
LIMIT 5;
```

| id        | token               |
|-----------|---------------------|
| 101934101 | callemondah         |
| 101933899 | beecher             |
| 85782291  | rockhampton central |
| 85775745  | west rockhampton    |
| 85775737  | rockhampton city    |


# Solving the problem

We've actually got a lot of tools for finding the child locations.
However the lineage is easy to use and does the job.
If we're looking for children down the lineage we need some way to sort them.
If what we're doing is related to people it makes sense to order by population, where it's available.
Otherwise we could assume area is the best proxy for likely location.

That finally leads to the original query.

```
SELECT
  docs.id,
  json_extract(json, '$.name') AS name,
  json_extract(json, '$.population') AS pop,
  json_extract(json, '$.geom.area') AS area
FROM lineage
JOIN docs ON docs.id = lineage.id
WHERE pid = 102049039
AND json_extract(json, '$.placetype') = 'locality'
ORDER BY pop DESC, area DESC
LIMIT 10;
```