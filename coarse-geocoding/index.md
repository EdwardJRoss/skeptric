---
categories:
- data
date: '2020-06-27T08:00:01+10:00'
image: /images/nominatim.png
title: Coarse Geocoding
---

Sometimes you have some description of a location and want to work out where it is.
This is called geocoding; if you just want to know what state or country it's in it's called coarse geocoding.
I found that while many structured JobPostings [contain a country](/sparql-job-country) some have it as a description rather than a country code, and some put the location in other fields.
We can often find the country using geocoding.

For example one job post has an `addressRegion` of ` Wales` and and `addressLocality` of `Dyfed`.
If I put this in the [Nominatim geocoder](https://nominatim.openstreetmap.org/search.php?q=Dyfed%2C+Wales&polygon_geojson=1&viewbox=) built on Open Street Map it gives a plausible location in the United Kingdom.
Similarly it can tell me that a job with `addressLocality` of `Regensburg` is in Bavaria, Germany.
It can also tell me the country `メキシコ` is Mexico.

This is great, but if I want to use it at scale to process my data they have a [usage limit of 1 request per second](https://operations.osmfoundation.org/policies/nominatim/).
While I could process the data slowly, or go to a [commercial provider](https://wiki.openstreetmap.org/wiki/Nominatim#Alternatives_.2F_Third-party_providers), I should be able to do it myself if it's build on Open Streetmap Data.

Nominatim can be [self-hosted](http://nominatim.org/release-docs/latest/admin/Installation/) but it requires 64GB of memory for an installation and setup including PostGIS.
This is because it's designed to be a performant address search engine at an address level.
Trying to batch normalise addresses to a country level with it is like using a sportscar to plough a field.

However the [placeholder](https://github.com/pelias/placeholder) library is a much better fit.
It's a component of the [Pelias geocoder](https://pelias.io/), but for geocoding at a regional level, and runs on a 2GB SQLlite database.
They have a [live demo](https://placeholder.demo.geocode.earth/demo/#eng) and a guide to [getting a coarse geocoder in (almost) one line](https://geocode.earth/blog/2019/almost-one-line-coarse-geocoding).
Another candidate is [twofishes](https://github.com/foursquare/fsqio/tree/master/src/jvm/io/fsq/twofishes) from foursquare, but they don't make it as easy to use.

It's actually really easy to get started and returns a list of JSON results which works accross multiple languages.
The top result is normally really good; here's an example of the output containing the 'lineage' of regions above it and a bounding box.
The ids refer to [Whos on First](https://whosonfirst.org/).


```
{'id': 85676997,
 'name': 'Dakar',
 'abbr': 'DK',
 'placetype': 'region',
 'population': 3137196,
 'lineage': [{'continent': {'id': 102191573,
    'name': 'Africa',
    'languageDefaulted': True},
   'country': {'id': 85632365,
    'name': 'Senegal',
    'abbr': 'SEN',
    'languageDefaulted': True},
   'region': {'id': 85676997,
    'name': 'Dakar',
    'abbr': 'DK',
    'languageDefaulted': True}}],
 'geom': {'area': 0.045757,
  'bbox': '-17.530915772,14.586814028,-17.1262208188,14.8863398389',
  'lat': 14.772684,
  'lon': -17.220068},
 'languageDefaulted': True}
```

On my laptop running in a docker container I can fetch and parse a result in around 50ms.
This is about 20 requests per second, so it would take around 14 days to get through all 1 million JobPosting URLs in the 2019 Web Data Commons extract.
However many locations are repeated and this could be reduced by an order of magnitude with some caching.