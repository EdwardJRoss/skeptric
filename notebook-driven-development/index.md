---
categories:
- ''
date: '2021-01-22T16:49:46+11:00'
draft: true
image: /images/
title: Notebook Driven Development
---

# End Note: Iterative development

In the early days of computer programming when compute was expensive you would carefully design your whole program (and perhaps prove them correct) before you started coding.
This is still done in some areas today where it's worth investing in formal specifications, like compilers or critical safety equipment like medical devices.
But in reality a lot of code is written without exactly knowing what you're trying to achieve, with changing requirements.

Another approach is test driven development where you start by writing tests with how your component is meant to act, before you implement them.
Again this is good when you know what you're doing.

I often have no idea what I'm doing.
In data analysis I am often consuming under-documented APIs, dealing with data sources that have issues I only discover as I go along, and have to tune and adapt rules to fit the data.
I often only need it to work in a certain predefined scope, as compared with say a mobile application which needs to run on a bunch of different devices in different conditions.
In this case iterative development works really well and helps me get started.

I started iterative development using a REPL; coding there and copying snippets back into my editor.
This is really inefficient and so I used emacs (with excellent modes like ESS for R and elpy for Python) to send code into a REPL and play with it as I developed it.
This works pretty well, but I would lose my results and start doing things like adding commented results for posterity (I never quite got [org babel](https://orgmode.org/worg/org-contrib/babel/) working which can do this better).
Jupyter Notebooks automatically keep the output, let you easily add prose and can render things like HTML and images, which are indispensable when parsing HTML or building image recognition algorithms.
While I miss some Emacs or IDE features in Jupyter notebooks they're still pretty good.