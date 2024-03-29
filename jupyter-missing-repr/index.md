---
categories:
- python
- jupyter
date: '2020-12-21T17:54:17+11:00'
image: /images/jupyter_repr.png
title: Fixing repr errors in Jupyter Notebooks
---

When running the Kaggle API method `dataset_list_files` in a Jupyter notebook I got an error about `__repr__` returning a non-string.
At first I thought the function was broken, but then I realised it was just how it was displaying in Jupyter that was breaking because the issues were all in `IPython`:

```
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
IPython/core/formatters.py in __call__(self, obj)
    700                 type_pprinters=self.type_printers,
    701                 deferred_pprinters=self.deferred_printers)
--> 702             printer.pretty(obj)
    703             printer.flush()
    704             return stream.getvalue()

IPython/lib/pretty.py in pretty(self, obj)
    392                         if cls is not object \
    393                                 and callable(cls.__dict__.get('__repr__')):
--> 394                             return _repr_pprint(obj, self, cycle)
    395 
    396             return _default_pprint(obj, self, cycle)

IPython/lib/pretty.py in _repr_pprint(obj, p, cycle)
    698     """A pprint that just redirects to the normal repr function."""
    699     # Find newlines and replace them with p.break_()
--> 700     output = repr(obj)
    701     lines = output.splitlines()
    702     with p.group():

TypeError: __repr__ returned non-string (type NoneType)
```

By *assigning* the output, rather than displaying it, the error goes away.
If I try to display the output I get the same error again.
By running `type` I can see that it's a `kaggle.models.kaggle_models_extended.ListFilesResult`, and by running `dir` I can see that the two non-special attributes are `error_message` and `files`.
The `files` contains the files I want and does have a working `repr`.
So I can monkeypatch the class:

```python
from kaggle.models.kaggle_models_extended import ListFilesResult
ListFilesResult.__repr__ = lambda self: repr(self.files)
```

Then displaying the result works just fine in Jupyter.