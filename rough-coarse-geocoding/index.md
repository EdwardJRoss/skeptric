---
categories:
- data
date: '2020-08-26T15:28:43+10:00'
image: /images/coarse_geocoding.png
title: Rough Coarse Geocoding
---

A [coarse geocoder](/coarse-geocoding) takes a human description of a large area like a city, area or country and returns the details of that location.
I've been looking into the source of the excellent [Placeholder](https://github.com/pelias/placeholder) (a component of the Pelias geocoder) to understand how this works.
The overall approach is straightforward, but it takes a lot of work to get it to be reliable.

A key component geocoder is a gazetteer that contains the names of locations.
Placeholder uses [Who's on First](https://whosonfirst.org/) which is a large open dataset that captures locations as GeoJSON based on how people describe them (including names in many languages).
The returned locations are Who's on First entities.
Placeholder stores these as tables in a SQLite database, which can be used to [refine locations in Placeholder](/placeholder-refining-location).

The overall approach of Placeholder is:

* Normalise text and expand synonyms
* Tokenise based on existing tokens
* Search for locations from general to specific
* Return results in order

So for example consider some text like "Saint Albans, Australia".
This gets normalised to "st albans australia".
This then gets tokenised to "st albans" and "australia".
Next "australia" is matched to the [country](https://spelunker.whosonfirst.org/id/85632793/).
Then "st albans" is searched for in Australia, and it finds a few results in Victoria and New South Wales.
Further it does an [R-tree](https://en.wikipedia.org/wiki/R-tree) search for "st albans" in locations within 2 degrees of Australia and finds another location in New Zealand.
These are then returned ordered by Who's on First id.


# Normalisation

Normalisation is the process of taking the input text and putting it in a format that makes it easy to match with Who's on First data.
This includes adding synonyms to expand the search.
This is really important in making the geocoder work in practice.

In Placeholder most of this work is done by the function `normalize` in `/lib/analysis.js`.
This function does a lot; I'll just show a few transformations to give an idea of what it is doing.


All the separating punctuation is stripped away.

```js
  // remove certain punctuation
  input = input.replace(/[\.]+/g,'');

  // replace certain punctuation with spaces
  input = input.replace(/[",]+/g,' ');
```

I'm guessing Who's on first tends to use contracted forms because Placeholder replaces e.g. "saint" with "st".

```js
  // generic synonym contractions
  input = input.replace(/\b(sainte)\b/gi, 'ste')
               .replace(/\b(saint)\b/gi, 'st')
               .replace(/\b(mount)\b/gi, 'mt')
               .replace(/\b(fort)\b/gi, 'ft');
```

The synonyms are actually a list because there can be multiple ways to describe a place.
For example if we have "city of sydney" it will try both "city of sydney" and "sydney".

```js
  // synonymous representations of official designations
  if (input.match(/county|city|township/i) ){
    synonyms = synonyms.concat(
      synonyms.map(synonym => {
        return synonym
          .replace(/^county\s(of\s)?(.*)$/gi, '$2')
          .replace(/^(.*)\scounty$/gi, '$1')
          .replace(/^city\sof(?!\s?the)\s?(.*)$/gi, '$1')
          .replace(/^(.*\s)charter\s(township)$/gi, '$1$2');
      })
    );
  }
```

Finally all text is converted to lowercase and unicode accents are removed.

These kind of transformations are really important for real world performance, but require a lot of experience to get right.
If you wanted to write your own geocoder based on Who's on First I'd seriously consider using their tests in `analysis.js`.

## Tokenize

The tokenisation is a little difficult in that place names can contain multiple words.
The approach in Placeholder, in `prototypes/tokenize.js`, is relatively simple.
First break the query into words, and start at the leftmost token.
Then take the span from the first to last word and if that's in the gazetteer then use that as the word, otherwise remove the last word from the span and repeat until you find a token or get down to a single word.
Then continue to tokenize the rest of the text.

For example consider "Port of Spain Trinidad and Tobago".
This isn't in the gazetteer, not if "Port of Spain Trinidad and", or "Port of Spain Trinidad", but "Port of Spain" is and so that's our first token.
Then to tokenize "Trinidad and Tobago" that is in our gazetteer and so is a token.
So we get two tokens "Port of Spain" and "Trinidad and Tobago".

As another example "Melbourne CBD Australia" tokenizes to "Melbourne CBD" and "Australia", since "Melbourne CBD" is in Who's on First.
But "Sydney CBD Australia" (currently) tokenizes to "Sydney", "CBD" and "Australia" since "Sydney CBD" is not in Who's on First.

This is a simple strategy but works pretty well.

## Search

In the west people normally write locations from specific to general, e.g. "Melbourne CBD, Victoria, Australia".
Although in some cultures, like Vietnamese, people write the opposite way from general to specific.
Placeholder assumes people are writing from specific to general and refines the search from general to specific.

So it will start with the rightmost token, in this case "Australia" and gets a list of locations.
It then uses the Who's on First lineage to look for "Victoria" in each "Australia" (or nearby to Australia using an R-tree search).
If it finds one it will refine to searching in Victoria for Melbourne, otherwise it will search for Melbourne in Australia.
Finally it returns all the results.

This heuristic works pretty well, but sometimes has odd results.
For example "Sydney CBD Australia" is tokenized to "Sydney", "CBD" and "Australia".
It so happens there is currently only one entity in Who's on First with a tag of CBD in/near Australia, and that happens to be [Melbourne CBD](https://spelunker.whosonfirst.org/id/85782343/).
So it then looks for Sydney in Melbourne CBD, but doesn't find one and returns Melbourne CBD.

On the other hand "Sydney CBD NSW Australia" first finds the state of "New South Wales" in "Australia".
Then it fails to find a "CBD" in "New South Wales", and so searches for "Sydney" in "New South Wales" and finds the capital.

## Sort

The sorting is very important, for example if I'm searching for "Paris" without any context I'm most likely to be searching for [Paris, France](https://spelunker.whosonfirst.org/id/85683497/) than [Paris, USA](https://spelunker.whosonfirst.org/id/101725293/).
However as far as I can tell Placeholder just sorts in order of the Who's on First id.
In practice this seems to work remarkably well; the larger and more populous places tend to occur first.
I don't know why this is; maybe it's because the dataset was built up starting with the most common places first.

## Putting it together

The Placeholder geocoder takes a relatively straightforward approach, but it's pretty effective.
I've been using it to [geocode Australian locations](/placeholder-australia) and it's really easy to use through docker.
However I'm finding I want to be able to customise it and make it fit better with the rest of my Python code.
I don't think it would be tremendously difficult to port to Python, although requires to be deliberate to get exactly the same results.
