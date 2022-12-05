---
categories:
- python
- programming
- legacy code
date: '2021-08-13T17:44:10+10:00'
image: /images/testing_pain_gain.png
title: Changing Python Analytics Code
---

> This is the essence of the refactoring process: small changes and testing after each change.
> If I try to do too much, making a mistake will force me into a tricky debugging episode that can take a long time.
> Small changes, enabling a tight feedback loop, are the key to avoiding that mess.
>
> Martin Fowler, *Refactoring, Improving the Design of Existing Code*

You've got a Python analytics process and have to make a change to how it works.
You [trace the dataflow](/dataflow-chasing) and work out how you need to restructure the code to make the change.
But how do you only make the change required without breaking something else?

A common technique I've used is to make the changes, run the pipeline, then compare diagnostics before and after.
Often these processes produce a report, or test set metrics or a dataset that can be compared.
If the process runs and the outputs aren't too different then I probably haven't broken anything.

However this process is typically very slow.
Analytics pipelines can take a long time to run, comparing the outputs can be very time consuming, and deciding whether the changes are too severe can be error prone.
Because it takes a long time to evaluate changes, it's not worth making small changes; the overhead in checking them is too large.
So a large change-set is made, evaluated, and then when there are errors a long time is spent tracing and debugging them.
This is a slow and frustrating process.

How can we shorten the feedback loop and make things faster?


# Unit Tests

Unit tests are the most effective tool to [make changes faster](/tests-make-changes-faster).
If the code you're changing is under test, you can make a small change and run the tests in a few seconds.
If the tests pass you'll get a lot of confidence that you haven't broken it, and can go on to the next change.
If they fail you know exactly where the error occurred and can quickly isolate it.

However what if the code isn't tested?
[Automatic refactoring](/automated-refactoring) tools can help to safely change the code.
In particular extract method is very useful for turning part of a large sequence of imperative statements into a separate function that can be tested.
You can then write the test before you make the change, and then make the change.

You may need to change and restructure tests as you change the code, but you can separate out making the change from checking it acts as expected. 

Explicit unit tests also make for great documentation.
Property based tests, using [Hypothesis](https://hypothesis.readthedocs.io/en/latest/), can test the code a lot more thoroughly, but tend to be a bit slower.
Bigger tests such as integration tests (which may do things like connect to the database, or hit APIs) will tend to be too slow to run on this cycle, but can be run intermittently.
Finally [regression tests](/diff-tests), seeing how the output has changed after a full run can give a lot more confidence, but it much slower.
It may be worth running this after all the small changes, that have passed the unit tests, to further confirm it acts as expected.

# Linters

Have you ever run a pipeline for a long time, only for it to fail halfway through because of a typo, or a missing import?
[Pylint](https://www.pylint.org/) and [pyflakes](https://github.com/PyCQA/pyflakes) are tools that can quickly check your code and catch many of these kinds of bugs.
Many Integrated Development Environments have these kinds of tools built into them, but they're fast enough it can be worth adding in a pre-commit hook or as a check in Continuous Integration.

The easiest way to run pylint to just catch errors is `pylint -E <path-to-module-or-script>`.
The warnings are often useful too, which can be added using `pylint --disable=R,C <path-to-module-or-script>`
It can also do style checking, but requires quite a bit of configuration because the defaults are way too strict.

Pyflakes is much faster than pylint, but generally seems to catch less errors.
If you're interested in checking style at the same time you can use it in [flake8](https://flake8.pycqa.org/en/latest/index.html).

These tools don't check a lot, but they can very quickly find issues so are worth having in the toolbox.

# Type Checking

Type checking is a useful tool to make sure your code *fits together*; that the right kinds of arguments are being passed.
It's easy when refactoring to make mistakes here that type checking can fix.

Even though Python has type hints, and ways to check them like [mypy](http://mypy-lang.org/) but they are of limited usefulness in code that used pandas and numpy heavily.
Unfortunately Pandas [doesn't support annotating dtypes](https://github.com/pandas-dev/pandas/issues/26766) which means a lot of real issues will be missed that have to be caught in tests.
You could wrap types like [thinc](https://thinc.ai/docs/usage-type-checking) to check things like dimensions in numpy, but it's still easy to make errors.

While more Python libraries are incorporating types, you may still need to write [type stubs](/python-type-stubs) to fill in missing types, which makes it even harder to get started with types.

Type checking is potentially worthwhile, and type annotations make good validation, once you've got a reasonable test suite and a linter in place.


# Contract validation

One of the challenges with data analytics code is that both the code and the data can change.
Even if you've got good tests to make sure the code works well, you can still get malformed data that you can only check at runtime to make sure you've got the right data.
This is where runtime validations, or contracts, come into account.

A contract is typically an assert statement in the code.
You may want to make sure the data coming in is as expected, or the data coming out is as expected.
It's better to fail fast if something is seriously wrong than to produce the wrong data.

One tool in this space is [pandera](https://pandera.readthedocs.io/en/stable/) which lets you verify the schema of a dataframe.
There's also [bulwark](https://github.com/zaxr/bulwark), and [tdda](https://tdda.readthedocs.io/en/v1.0.32/index.html) that tries to discover the schema; more generally there is [pydantic](https://pydantic-docs.helpmanual.io/) and [marshmellow](https://marshmallow.readthedocs.io/en/stable/) for checking data against a schema.
For databases there's [great expectations](https://greatexpectations.io/) but I'm not convinced that it's better than just writing some checks by hand.


# Getting faster feedback cycles

I've generally found the faster I find problems the faster I can change them.
The more confidence I have that I will catch errors, the more aggressively I will change my code.
Python by default doesn't have much of a safety net, but with tests, linters, type checking and contract validation you can fail faster and be more confident the changes you've made haven't broken anything.