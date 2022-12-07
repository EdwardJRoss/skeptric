---
categories:
- programming
- blog
date: '2020-10-31T21:29:34+11:00'
image: /images/jamstack.png
title: Choosing a Static Site Generator
---

Static website generators fill a useful niche between handcoding all your HTML and running a server.
However there's [a plethora](https://jamstack.org/generators/) of site generators and it's hard to choose between them.
However I've got a simple recommendation: if you're writing a blog use Jekyll (if you don't want to use something like Wordpress).

Static website generators compile input assets into a set of static HTML, CSS and Javascript files that can be deployed almost anywhere.
They can do useful things like render formatting in pages, create indexes, RSS feeds, optimise images and minify assets.
However different generators are built in different programming languages, have different features and conventions for assets, and different plugins.

I'm going to say if you don't know what to choose for a blogpost go with Jekyll.
It's one of the most popular generators, and one of the oldest (older than the Javascript frameworks used for many popular generators).
This means that it's featureful, stable, has a huge community and consequently tons of themes and plugins.
It's written in Ruby which is a dynamic language making it relatively easy to write plugins.

This site is currently using Hugo through a [Github Action](/github-actions).
I've also used it to publish [Jupyter notebooks](/jupyter-hugo-blog) and [R Blogdown posts](/blogdown).
However Hugo makes breaking changes, which forced me to [change my theme](/casper-2-to-3) and doesn't have an easy way to render diagrams.

It would be great to be able to put declarative diagrams inline with the article.
Unfortunately it looks very [unlikely Hugo will support PlantUML](https://github.com/gohugoio/hugo/issues/796); someone would have to port it to Go.
I ended up resorting to using [Mermaid for diagrams](/diagrams-in-hugo) but client side rendering makes it much slower to load and paint the page (it's a major factor according to Google site tools).
This makes it a less pleasant experience and hurts SEO.
Jekyll will be much slower to generate the website than Hugo, but with something like the [Jekyll PlantUML](https://github.com/yegor256/jekyll-plantuml) it could generate a website that's much faster to load.
Another way to do this would be with Blogdown, which renders the pages using RMarkdown which executes code in R then uses Pandoc to convert the output, and hands off to Hugo (or other generators) to make related assets from the HTML output.

I'm not sure if it's worth me switching to Jekyll now with it's [wide array of plugins](https://github.com/planetjekyll/awesome-jekyll-plugins) and [themes](https://jekyllthemes.io/).
I'll have to investigate the tradeoffs more, and compare it to filling the gaps with RMarkdown.
But I regret picking Hugo over the more stable, popular and featureful Jekyll.
I'm not the only person who has had issues with Hugo, as can be seen on this [Hacker news thread](https://news.ycombinator.com/item?id=24945299).
