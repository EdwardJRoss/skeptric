---
categories:
- jupyter
- python
date: '2020-08-07T08:00:00+10:00'
image: /images/jupyter_latex_html.png
title: How to turn off LaTeX in Jupyter
---

When showing money in Jupyter notebooks the dollar signs can disappear and turn into LaTeX through Mathjax.
This is annoying if you really want to print monetary amounts and not typeset mathematical equations.
However this is easy to fix in Pandas dataframes, Markdown or HTML output.

For Pandas dataframes this is especially annoying because it's much more likely you would want to be showing \$ signs than displays math.
Thankfully it's easy to fix by setting the [display option](https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html) `pd.options.display.html.use_mathjax = False`.
It's strange this is True by default, but you can add this configuration near the top of all your Jupyter notebooks.

Unfortunately you can't turn it off in Markdown; you'll just need to replace every `$` sign with `\$`.
The backslash turns off the LaTeX rendering and you'll just get dollar signs.
A little annoying to remember, but straightforward to work around.

I often display HTML in Jupyter notebooks using [`IPython.display.HTML`](https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.HTML), and using [`IPython.display.display`](https://ipython.readthedocs.io/en/stable/api/generated/IPython.display.html#IPython.display.display) as the equivalent of print.
Unfortunately the HTML also interprets dollar signs as Mathjax, but again you can turn this off by escaping the dollar signs with a backslash.
Since this is what you will almost always want you can wrap it in a function:

```python
from IPython.display import display, HTML as HTML_raw

def HTML(text):
  text = text.replace('$', r'\$')
  return HTML_raw(text)
```

This doesn't handle the ability to pass URLs or filenames, but is easy to extend to these cases.

The default of interpreting dollar signs as Mathjax in Jupyter output can be annoying and confusing, but once you know the problem it's straightforward to solve with a little extra configuration.