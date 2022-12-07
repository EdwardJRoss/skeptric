---
categories:
- programming
date: '2021-01-08T22:16:54+11:00'
image: /images/test_frame_iloc_fails.png
title: Hurdles in Contributing to Open Source
---

Often in programming it's not the code itself that is hard, it's all the environment and systems around it.
I found that today when trying to contribute to an open source repository.

Today I was working on some code and using the excellent [data-science-types](https://github.com/predictive-analytics-lab/data-science-types) to type check some Pandas code with `mypy`.
But for some reason I was getting a weird error when reading with `read_feather` some data I just wrote with `to_feather`, and so I switched my `to_feather` to be `to_pickle` which doesn't do as much conversion.
This worked fine but then `mypy` had an error:

```
error: "Series[Any]" not callable
```

It must have thought that `df.to_pickle` was the name of a column, because it wasn't in the type stub.
Well through the wonders of open source I can easily fix that, I [opened an issue](https://github.com/predictive-analytics-lab/data-science-types/issues/216), cloned the repository and as per the instructions installed it in a virtualenv with `pip install -e ".[dev]"` and ran the tests with `./check_all.sh` (it's great that they are clear and make it easy to get set up and run the tests).
But then I ran into an issue; one of the Pandas tests fails before I've even changed the code.

I see that it's using Pandas 1.2 which just came out in the last 2 weeks, so I install the previous version of Python using `pip install "pandas<1.2"` (note the quote; I keep forgetting it and my shell tries to do input redirection), and run the tests and sure enough it passes.
I [open an issue](https://github.com/predictive-analytics-lab/data-science-types/issues/215) about the failing tests.
I spent some time trying to work out what the test was and why it was failing, but I couldn't get to the bottom of it after half an hour or so, and so move on to the changes.

I make the changes and they're relatively straightforward, and I set up a [pull request](https://github.com/predictive-analytics-lab/data-science-types/pull/217).
However the CI tests fail (but only in the Python 3.9 environment, not in Python 3.6 which must have got an earlier version of Pandas) because of the issue I had.
The maintainer agrees we can workaround by limiting Pandas to <1.2 for now until the tests are fixed.
Luckily I'm familiar with Github actions [from publishing this website](/github-actions) and so I quickly see how to make this change and get the tests passing.

At the end of the day a straightforward change ends up taking a couple of hours because I had to work around an unrelated test was failing due to a change in another package.
This could have been caught earlier, at the time of the upstream change, if the tests ran regularly (say weekly).