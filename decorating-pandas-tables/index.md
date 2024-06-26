---
categories:
- python
- data
- pandas
- jupyter
date: '2020-11-22T20:43:20+11:00'
image: /images/pandas_df_heatmap.png
title: Decorating Pandas Tables
---

When looking at Pandas dataframes in a Jupyter notebook it can be hard to find what you're looking for in a big mess of numbers.
Something that can help is formatting the numbers, making them shorter and using graphics to highlight points of interest.
Using Pandas `style` you can make the story of your dataframe standout in a Jupyter notebook, and even export the styling to Excel.

The [Pandas style documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html) gives pretty clear examples of how to use it.
When you have your final dataframe you can then call `.style` and chain styling functions.

For example you can colour cells by their value using [`style.background_gradient`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.io.formats.style.Styler.background_gradient.html) to get an effect like Excel's *Colour Scales* Conditional Formatting.
You can choose a colormap through the cmap argument, using the [Matplotlib colormaps](https://matplotlib.org/tutorials/colors/colormaps.html).
One handy trick is to get the reverse of a colormap by appending `_r` to the name.

```python
(
 df
 .style
 .background_gradient(cmap="PuRd_r")
)
```

![Heatmap Dataframe](/images/pandas_df_heatmap.png)

You can even make a data barchart inside the dataframe using [`style.bar`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.io.formats.style.Styler.bar.html).
You can set the color, minimum and maximum values, axis and choose a subset of columns to show bars on.

```python
(
 df
 .style
 .bar(vmax=len(df), color='lightblue')
)
```


![Data bar chart in Dataframe](/images/pandas_bar_chart.png)

To make the data easier to read you can add a [`style.format`](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.io.formats.style.Styler.format.html).
This can take a dictionary of columns to formatters which can be [format strings](https://docs.python.org/3/library/string.html#format-specification-mini-language) or functions.
Because the HTML is rendered you can actually use this to do things like put in decorations.

```python
def format_arrow_text(value):
    if value < 0:
        indicator = '<span style="color:red;">⮟</span> ' 
    elif value > 0:
        indicator = '<span style="color:green;">⮝</span> ' 
    else:
        indicator = ''
    return f'{indicator} {value:.1%}'
    
df.style.format(format_arrow_text)
```

![Example of Format With Arrows](/images/pandas_df_format.png)

This is just scratching the surface, you can do a lot more by writing custom styles.
It's convenient for simple things but styling with just CSS attributes at a cell level is a bit clunky, and for complex things you'll want to render your own HTML (potentially by [subclassing](https://pandas.pydata.org/pandas-docs/stable/user_guide/style.html#Subclassing)).

For contrast R has the [formattable package](https://github.com/renkun-ken/formattable) which can achieve many of the [same things](https://cran.r-project.org/web/packages/formattable/vignettes/formattable-data-frame.html).
In this case the syntax isn't much better than Pandas.