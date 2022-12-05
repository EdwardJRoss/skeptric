---
categories:
- r
- python
date: '2020-10-24T22:13:54+11:00'
image: /images/r-python.jpg
title: 'R: Keeping Up With Python'
---

About 5 years ago a colleague told me that the days were numbered for R and Python had won.
From his perspective he is probably right; in software engineering companies Python has got increasing adoption in programmatic analytics.
However R has its own set of unique strengths which make it more appealing for the stats people and has kept up surprisingly well with Python.

Python has a wider audience than R, and keeps to its reputation as "not the best language for anything but the second best language for everything".
It may be the best language for Deep Learning; both Tensorflow and PyTorch are developed primarily for Python.
The package ecosystem for Python is growing quickly.

However R aims squarely at analysts and statisticians, who are not as close to the low level details but are typically closer to the problems to solve.
A strength of this is there is a lot more focus in the R community on usability.
Dplyr, and its derivatives such dbplyr, give a clear consistent way of manipulating data; compared with the hodge podge of Pandas (which is closer to base R), SQLAlchemy, and PySpark.
RMarkdown and its derivatives blogdown and bookdown make it really easy to integrate text and executed code in a way that's much more difficult than Sphinx and produces more reproducible and aesthetic output than Jupyter Notebooks.
Ggplot2 makes it much easier to produce and iterate on bespoke graphics than Matplotlib and Seaborn (Altair is closer but can't produce graphics directly since it builds on Vega-lite).
Shiny makes it easy to build an interactive analytics application; I'm not sure whether Dash is up to scratch now but the alternative in Python is building your own Django application which requires much more expertise.
I suspect for model fitting R is more versatile than Python's scikit-learn (I'll know after I read [Tidy Modeling With R](https://www.tmwr.org)).

R is also doing admirably filling in the gaps where Python is ahead.
In particular reticulate lets you use Python from R, which can cover many gaps (and there's rpy2 for calling R from Python).
The renv package helps with the pain point of environment management, catching up with Python's pip and venv (and it can manage Python dependencies too for cross-language projects).
There's an interface to Tensorflow, and now to Torch in R.

While I'd still stick to Python for most production analytics workloads, R is doing an admirable job.
As an analytics tool R is more powerful and much easier to use; if it keeps leaning into these strengths it will continue to be popular for a long time.