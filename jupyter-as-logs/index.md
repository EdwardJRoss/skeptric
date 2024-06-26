---
categories:
- jupyter
- python
date: '2021-02-06T08:00:00+11:00'
image: /images/jupyter_nbconvert.png
title: Jupyter Notebooks as Logs for Batch Processes
---

When creating a batch process you typically add logging statements so that when something goes wrong you can more quickly debug the issue.
Then when something goes wrong you either try to fix and rerun it, or otherwise run the process in a debugger to get more information.
For many tasks Jupyter Notebooks are better for these kinds of batch processes.

Jupyter Notebooks allow you to write your code sequentially as you usually would in a batch script; importing libraries, running functions and having assertions.
But additionally the outputs can be displayed directly in the notebook, without needing to resort to separate log statements.
Moreover the outputs can be more than text; they can be images (such as graphs) or rich HTML markup or even [interactive elements](https://www.nbinteract.com).
When you run into an issue you can just interactively step through the notebook, which gives you a much richer experience than `pdb`.

You want the notebook to be like a script and not contain the output, so before committing it to source control you want to clear all the outputs (and may want to set up a git hook for this):

```sh
jupyter nbconvert --inplace --clear-output batch_script.ipynb
```

Then using nbconvert you can execute the notebook, and render it as a HTML logfile.
For example this runs the notebook, removes the prompts (the `[1]`, ...) and outputs it to a logfile with the UTC run date.

```sh
jupyter nbconvert batch_script.ipynb --to html --execute \
  --output logs/`date --utc +%Y%m%dT%H%M%SZ`_batch_script.html \
  --no-prompt
```

Of course sometimes you'll want logging statements from the libraries you import, and you can display them in your Jupyter notebooks:

```python
import logging
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)
```

If you want to go further and parameterise your scripts you could read in a config file in the notebook, or use something like [Papermill](https://papermill.readthedocs.io/en/latest/index.html) to pass the parameters.