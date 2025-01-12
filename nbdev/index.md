---
categories:
- jupyter
- python
date: '2021-01-22T19:12:15+11:00'
image: /images/nbdev_docs.png
title: Getting Started with nbdev
---

[Nbdev](https://nbdev.fast.ai/) is a tool to make it possible to develop Python libraries in Jupyter notebooks.
At first I found this idea scary, but after watching the talk [I like Notebooks](https://www.youtube.com/watch?v=9Q6sLbz37gk&feature=youtu.be) and seeing how it works I think it's got the best of all worlds.

It lets you put code, documentation, examples and tests all together in context and provides tooling to extract the code into an installable library, run the tests and produce great hyperlinked documentation.
In this sense it's reminiscent of Knuth's WEB System for literate programming, in which you would write the text and code together and there was a command to `TANGLE` compilable code or `WEAVE` TeX documentation from the source.
However it's much richer than WEB because Jupyter notebooks provide an interactive live coding environment, and can embed HTML, images and other rich media.
It's natural in Jupyter notebooks to see what happens when you run a piece of code, and with nbdev this then becomes an example in the documentation and optionally a test you can run.

There are lots of rough edges with making Jupyter notebooks reproducible and easy to diff, like the order of execution and numbers in cells.
Nbdev provides tools that make all this easy, and allows you to sync code *two ways* between notebooks and python source files.
This means you can make code changes in an IDE and then sync it back to the notebooks.
However there are still some limitations here (especially being consistent with the documentations) and I suspect a large refactor would be considerably harder.


# Literate Code

One of the biggest benefits of nbdev is the documentation.
When I ran through the tutorial it [created documentation from my Jupyter Notebooks](https://edwardjross.github.io/nbdev_tutorial/#How-to-use) which is beautifully cross-referenced and displays images, HTML and the like without any extra effort.
A big hurdle when dealing with a new codebase is understanding what it does and how to use it.
Having documentation filled with examples and links back to the source is an excellent experience, and because the docs are kept with the code in the notebook (and tested together) they're much more likely to stay in sync than separately maintained documentation.

Nbdev itself is written in nbdev and you can see some of the limitations in the approach.
While library functions are reasonably well documented, configuration and command line scripts aren't that well documented.
I ran into a few problems running nbdev with the settings file, and syncing code back to the notebook.

## settings.ini

There's a `settings.ini` file that contains fields used for generating the documentation, setup.py and some other things.
In the template there are a bunch of fields commented out and it's not immediately obvious which ones you need to fill in, and what you should put there.

The first time I ran `nbdev_build_lib` to tangle the code from the notebooks I got `KeyError: 'copyright'`.
I then uncommented and updated the copyright setting to `copyright = 2021 Edward Ross`, only to see in the documentation:

> Copyright: 2021 Edward Ross 2021

Clearly I should have just set the field as `copyright = Edward Ross` to get the correct output

> Copyright: Edward Ross 2021

Then when I tried to install the package from the generated `setup.py` with `pip install .` I got this stack trace

```
    ERROR: Command errored out with exit status 1:
    ...

    AssertionError: missing expected setting: keywords
    ----------------------------------------
ERROR: Command errored out with exit status 1: python setup.py egg_info Check the logs for full command output.
```

Apparently I did need to set the `keywords` field (I had no idea what to put there, I thought it was just for publishing on pypi) and a `description`.

With these fields set it worked, but it would have been easier if I had more guidance over what these fields should be and which ones were required.

# Syncing code back to the notebook

I'm really interested by the idea of being able to sync code back to the notebook.
For some tasks it's easier using the tooling in an IDE or a command line utility than editing notebooks.
Unfortunately I found it a little hard to use.

Reading [the documentation](https://nbdev.fast.ai/sync.html#nbdev_update_lib) I tried just running `nbdev_update_lib` (I guess in WEB this would be `UNTANGLE`)

```
> nbdev_update_lib
Traceback (most recent call last):
  File "nbdev_update_lib", line 8, in <module>
    sys.exit(nbdev_update_lib())
  File "fastcore/script.py", line 105, in _f
    tfunc(**merge(args, args_from_prog(func, xtra)))
  File "nbdev/sync.py", line 125, in nbdev_update_lib
    if fname.endswith('.ipynb'): raise ValueError("`nbdev_update_lib` operates on .py files.  If you wish to convert notebooks instead, see `nbdev_build_lib`.")
AttributeError: 'NoneType' object has no attribute 'endswith'
```

This is an obtuse error, I had no idea what was causing it.
Maybe I need help?

```
> nbdev_update_lib --help`

usage: nbdev_update_lib [-h] [--fname FNAME] [--silent SILENT]

Propagates any change in the modules matching `fname` to the notebooks that created them

optional arguments:
  -h, --help       show this help message and exit
  --fname FNAME    A python filename or glob to convert
  --silent SILENT  Don't print results (default: False)
```

Even though `--fname` is an optional argument, the script fails if you don't pass it!
I tried a few different ways of writing the filename, such as `core.py`, but they all silently finished without changing the notebook.

I ended up looking at the source to see what happened:

```
files = nbglob(fname, extension='.py', config_key='lib_path')
```

It sounds like some sort glob, and I guess it fails if it's None (the default).
I tried passing "*" to get all files and it worked:

`nbdev_update_lib --fname "*"`
