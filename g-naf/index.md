---
categories:
- data
date: '2020-04-30T22:05:00+10:00'
image: /images/g-naf.png
title: Locating Addresses with G-NAF
---

A very useful open dataset the Australian Government provides is the [Geocoded National Address File (G-NAF)](https://data.gov.au/dataset/ds-dga-19432f89-dc3a-4ef3-b943-5326ef1dbecc/details).
This is a database mapping addresses to locations.
This is really useful for applications that want to provide information or services based on someone's location.
For instance you could build a custom store finder, get aggregate details of your customers, or locate business entities with an address, for example ATMs.

There's another open and editable dataset of geographic entities, [Open Street Map](https://www.openstreetmap.org/) (and it has a pretty good open source Android app [OsmAnd](https://osmand.net/)).
Unfortunately the G-NAF data [can't be used](https://lists.openstreetmap.org/pipermail/talk-au/2016-June/010961.html) in Open Street Map because it has a restriction (that you need to verify an address before you send mail to it) that's incompatible with Open Stree Maps licence.
This is really annoying because there are lots of gaps in Open Street Map's addresses which makes it difficult to use for navigation.
Though it is possible to [import to a local instance](https://help.openstreetmap.org/questions/63105/updating-osm-database-with-government-data-sets).
I'm still not sure how to use this effectively in OsmAnd.

The G-NAF data is a bit convoluted to use directly but there exists code to [load into Postgres](https://github.com/minus34/gnaf-loader) or even [use it in a web interface](https://github.com/data61/gnaf).
I'm really impressed that the government made this open and that it's regularly updated.