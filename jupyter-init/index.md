---
categories:
- jupyter
- python
- r
date: '2021-02-05T18:54:41+11:00'
image: /images/jupyter_preamble.png
title: Jupyter Notebook Preamble
---

Whenever I use Jupyter Notebooks for analysis I tend to set a bunch of options at the top of every file to make them more pleasant to use.
Here they are for Python and R with IRKernel

## Python

```python
# Automatically reload code from dependencies when running cells
# This is indispensible when importing code you are actively modifying.
%load_ext autoreload
%autoreload 2

# I almost always use pandas and numpy
import pandas as pd
import numpy as np

# Set the maximum rows to display in a dataframe
pd.options.display.max_rows = 100
# Set the maximum columns to display in a dataframe
pd.options.display.max_columns = 200
# Set the maximum width of columns to display in a dataframe
pd.options.display.max_colwidth = 80
# Don't render $..$ as TeX in a dataframe
pd.options.display.html.use_mathjax = False
```

## R

For R I configure similar display options to Python through [repr](https://irkernel.github.io/docs/repr/0.9/repr-options.html):

```R
# Set the maximum number of columns and rows to display
options(repr.matrix.max.cols=150, repr.matrix.max.rows=200)
# Set the default plot size
options(repr.plot.width=18, repr.plot.height=12)

# Usual analysis libraries
suppressPackageStartupMessages({
library(tidyverse)
library(ggformula)
library(glue)
library(lubridate)
}

# Database Libraries
library(DBI)
library(dbplyr)
```