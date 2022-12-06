---
date: '2020-07-26T08:00:00+10:00'
image: /images/casper2to3.png
title: Hugo Casper 2 to 3
---

I've been wanting to upgrade my version of Hugo, but the Casper 2 theme I was using didn't support it.
As a first step to this transition is to use [Casper 3](https://github.com/jonathanjanssens/hugo-casper3).
It looks similar to my old theme, is easy to set up, but seems to be missing some features.

I cloned the repository, and changed the theme in my `config.toml` to `theme = "hugo-casper3"`.
The article images weren't showing because the Casper 3 theme uses `feature_image` instead of `image` and requires a leading slash in the path (which was optional in 2).
I fixed both of these on my mmark articles using:

```
find . -name '*.mmark' -exec sed -i -E 's|^image( ?[=:] ?)"/?images|feature_image\1"/images|' {} \;
```

# Testing features

To make sure the conversion went well I created a list of articles using different features:


| Article                                                               | Image | Code | File | List | Quote | Table | LaTeX | RMarkdown |
|-----------------------------------------------------------------------|-------|------|------|------|-------|-------|-------|-----------|
| [Extracting Skills with Conjugations](/extract-skills-3-conjugations) | ✓     | ✓    | ✓    | ✓    |       | ✓     |       |           |
| [JobPosting Schema](/schema-jobposting)                               |       | ✓    |      |      | ✓     | ✓     |       |           |
| [Real Roots of Polynomials](/real-roots-of-polynomials)               | ✓     |      |      | ✓    |       |       | ✓     |           |
| [Blogdown](/blogdown)                                                 | ✓     | ✓    |      |      | ✓     |       |       |           |


The result of my tests:

* Images, files and links are working
* Syntax highlighting now works
* Quotes work as before, lists look better (the CSS was broken before)
* LaTeX was broken
* RMarkdown files lost their images

I've also lost a lot of features from the list page such as pagination, contact details and a cover image.
The theme doesn't look well supported, and I will investigate other themes in the future.
But this is a start and I can work on remediating LaTeX and RMarkdown.
