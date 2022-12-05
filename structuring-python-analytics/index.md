---
categories:
- python
- data
- legacy code
date: '2021-07-05T08:00:00+10:00'
image: /images/pipeline_tree.png
title: Structuing Python Analytics Codebases
---

Many analytics codebases consist of a pipeline of steps, doing things like getting data, extracting features, training models and evaluating results and diagnostics.
The best way to structure the code isn't obvious and if you're having trouble importing files, getting module not found errors or are tinkering with PYTHONPATH it's likely you haven't got it right.

A way I've seen many data analytics processing pipelines structured is with a series of numbered steps:

```
 src
 ├── 01_fetch_data.py
 ├── 02_extract_data.py
 ├── 03_normalise_data.py
```

The good thing about this structure is that the steps involved are clear with their order.
The bad thing about this is that you can't easily import these because they start with a number (but it can be done with [importlib](https://docs.python.org/3/library/importlib.html)), and they will only work when called from the right directory (otherwise paths will be messed up [without some magic](https://stackoverflow.com/questions/918154/relative-paths-in-python)).

Instead I recommend having each of the scripts in a directory with the name of the submodule (e.g. `my_pipeline`).
Then there are a few options on how to orchestrate it:

1. Making the module executable
2. Creating individual step scripts
3. Makefile or shell script orchestration

## Making the module executable

If you add a `__main__.py` (as well as the `__init__.py`) then you can execute the code in that file by running `python -m my_pipeline`.
You can then import the downstream steps there and set up a CLI library such as [Invoke](http://docs.pyinvoke.org/en/stable/), [Click](https://click.palletsprojects.com/), or [Typer](https://typer.tiangolo.com/).

```
 my_pipeline
 ├── __init__.py
 ├── __main__.py
 ├── fetch_data.py
 ├── extract_data.py
 ├── normalise_data.py
```
 
The nice thing about this is you don't need to worry about paths and imports.
It also lets you do more complex orchestration in the `__main__.py` file, and add other useful utilities that don't fit into the pipeline metaphor.

Another advantage of this method is you can store paths to intermediate steps in `__main__.py`.
If the [pipeline caches intermediate results](/caching-pipelines) in some sort of storage then each step needs to know where the last step put it.
This leads to duplication between the steps, or a large configuration file containing all the intermediate locations.
Instead we can make the input and output locations arguments in the scripts, and set them all in `__main__.py`.

Note that all your imports will need to be absolute to the package name, e.g `import my_pipeline.fetch`.

## Creating individual step scrtips

If you really like the idea of numbered scripts another way to handle it while having the right module structure is with symlinks.
The structure we end up with is like this:

```
 my_pipeline
 ├── __init__.py
 ├── fetch_data.py
 ├── extract_data.py
 ├── normalise_data.py
 scripts
 ├── 01_fetch_data
 ├── 02_extract_data
 ├── 03_normalise_data
```

The module level at the top level and removing the numbers from the scripts fixes the import and path problems.
Each of the scripts is a small shell script that invokes the corresponding step with `python -m my_pipeline.{step}`.

For example `01_fetch_data` might look like:

```shell
#!/bin/sh
/usr/bin/env python -m my_pipeline.fetch_data
```

This script is then invoked using `./scripts/01_fetch_data`.

You can invoke any of the individual scripts by passing the path with `-m`, e.g. `python -m my_pipeline.fetch_data`.
We then put all the scripts to run in a `scripts` subfolder with the numbers like this:

The drawback with this method is you need a way to deal with the intermediate state.

## Creating a Makefile or shell script

An alternative to using the `__main__.py` for orchestration is creating a Makefile or shell script.
The benefit of that is it can include functionality like installing a virtual environment or pulling a Docker image to run the Python scripts in (whereas the Python scripts won't be able to run without the environment).
The downside is these languages are a bit more brittle and less flexible than Python, and take more work to be nice to use than something like Typer.

```
 my_pipeline
 ├── __init__.py
 ├── fetch_data.py
 ├── extract_data.py
 ├── normalise_data.py
 Makefile
```

In this case I'd recommend making the input and output directories command line arguments that can be passed by the Makefile or shell script.
It can then run, for example the `extract_data`, using 

```sh
/usr/bin/env python -m my_pipeline.extract_data ./data/01_raw ./data/02_primary
```

It's critical you have a good command line interface with help (e.g. like [this for Make](https://gist.github.com/prwhite/8168133)) and good parsing.
A Makefile has the advantage that it can see if the intermediate dependencies have been built and automatically resolve what needs to be done.
However it requires some effort to get the sources and targets right so it works smoothly, and some targets might need to be generated for steps without output (e.g. setting up the environment).

# Making a choice

These are a few ways to structure your Python analytics pipeline with different tradeoffs.
In each of the cases we made a proper submodule where the steps could be separately run, which makes it easier to use the tools without thinking too much about paths.
In general there is conflicting advice around Python packaging, for example pypa [recommends putting the package inside the src/ directory](https://github.com/pypa/sampleproject), but others [recommend against an src/ directory](https://docs.python-guide.org/writing/structure/) since it makes it easier to run without installing.