---
categories:
- programming
- hugo
date: '2020-10-26T08:44:14+11:00'
image: /images/hugo_casper3_edit.png
title: Learning Hugo by Editing Themes
---

One of the hardest parts of learning something new is motivation.
This is why one of the best ways to [learn programming is editing code](/teaching-programming-by-editing); it's goal driven so motivation is built in.
I've successfully used this to start learning how to write Hugo themes.

Now that I've got a reasonable collection of posts, over 250, I would like to understand what content people are actually accessing on this website to get an idea of what would be useful.
Because I'm lazy the easiest way to do this is with Google Analytics.
Unfortunately the theme I'm using doesn't support Google Analytics, and I can't inject it into an existing partial like I did to [include diagrams in Hugo pages](/diagrams-in-hugo).

So I ended up forking the [Hugo Casper 3](https://github.com/jonathanjanssens/hugo-casper3) theme to create my own version where I could inject these tags.
I wanted to inject an appropriate partial hook so that I could put in the Google Analytics script; the existing `site-header.html` that I used to inject Mathjax and Mermaid.js isn't included on the list page.
There's an HTML template in `layouts/_default/baseof.html` that seems to contain the HTML template used to create every page, which would be appropriate.
This was as simple as adding the partial at the end of the head:

```
<head>
    ...
    {{- partial "header-scripts.html" $ -}}
</head>
```

Then I added a dummy `layouts/partials/header-scripts.html` in the theme:

```
<!--Configure scripts for headers here-->
```

Then in my website, which has the theme as a submodule, I overwrite `layouts/partials/header-scripts.html` with my Google Analytics tags (and also migrate my Mathjax and Mermaid code over to it).
Once I update the submodule in git it all works.

## Why not write your own theme?

The best thing about editing the theme is it opens the door to a lot of customisation.
It's easier than writing from scratch, and allows more flexibility than any theme.

I could write a theme from scratch, but that's a lot of work.
Not only would I have to understand HTML and CSS (and design) well enough to make a good website, I would need to understand Hugo well enough to do it.
I don't have the motivation to learn this whole framework from scratch; it's surprisingly complex for something that seems simple.

## Why not use an existing theme?

One of the appeals of Hugo is the vast array of [themes](https://themes.gohugo.io/) you can just drop in.
But unfortunately there's no consistency built into features in Hugo.
It was difficult when I [ported from Casper 2 to Casper 3](/casper-2-to-3) because the parameters were slightly different (and in fact I just realised today my social links had been broken since the upgrade 3 months ago).
For example Casper 2 uses `image` for a post image, others use `feature_image` (and others don't support it), Casper 2 used `githubName` to link to a github repository Casper 3 uses `github` and requires the full URL.

Moreover I couldn't find a single theme that did everything I wanted.
The closest was [academic](https://themes.gohugo.io/academic/), but it was slow to render and required a lot of configuration which has its own steep learning curve.


## Editing Themes

The wonderful thing about open source code is you can build on it.
Now I've opened the door to editing a theme the Hugo themes turns from a menu to a buffet.
I can take elements from other themes and integrate them into the theme I'm working on.

I can now start to look into making improvements to this blog, and slowly learning features of Hugo to solve real problems.
I've already sped up page loads by moving Mermaid.js to the end of the body instead of in the head.
There's more potential improvements by only loading the script on pages that use it (or pre-building the images), optimising image sizes, paginating the index page and so on.