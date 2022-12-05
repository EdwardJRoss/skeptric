---
categories:
- python
date: '2020-12-17T21:33:25+11:00'
image: /images/tox.png
title: Why use Tox for Python Libraries
---

I have been surprised how hard it is to maintain an internal library in Python.
There are constantly issues for end users where something doesn't work.
It turns out one feature used was introduced in Python 3.8, but someone was stuck on Python 3.6.
Changes to Pandas and PyArrow meant some combinations of those libraries broke.
It's really hard to build confidence in your system when lots of people end up with breakages.

A Python library needs to be tested on multiple versions of Python and dependencies.
The best way to get confidence that a program is working, and continues to work as you change it, is to write good tests of its functionality.
When you've got a Python application you can just pin the versions of everything (e.g. using [pip-tools](https://github.com/jazzband/pip-tools)), specify a version of Python, and both deploy and test in a controlled environment, or wrap it in a container.
For a library you have to support multiple versions of dependencies (otherwise no two packages would be compatible since they would be locked to different versions), and different versions of Python (because some users would be stuck on an old version and some on a new).
Obviously you can't support *every* possible combination, but for the dependencies and versions that are most likely to cause issues (fast changing things) you can support and test a few major versions.

[Tox](https://tox.readthedocs.io/en/latest/) is a really useful tool for testing a library on multiple *combinations* of Python versions and dependency versions.
It handles setting up different virtual environments with the correct set of dependencies and running the tests across all of them.
This is good because it takes quite a bit of discipline to set them up manually and check them; but you want tests to be easy and reliable to run.

Here's an example [from the tox documentation](https://tox.readthedocs.io/en/latest/example/basic.html#compressing-dependency-matrix).
You can specify different sets of dependencies that are combined together in a test matrix.

```
[tox]
envlist = py{36,37,38}-django{22,30}-{sqlite,mysql}

[testenv]
deps =
    django22: Django>=2.2,<2.3
    django30: Django>=3.0,<3.1
    # use PyMySQL if factors "py37" and "mysql" are present in env name
    py38-mysql: PyMySQL
    # use urllib3 if any of "py36" or "py37" are present in env name
    py36,py37: urllib3
    # mocking sqlite on 3.6 and 3.7 if factor "sqlite" is present
    py{36,37}-sqlite: mock
```

Even if you are only supporting a very limited set of versions it can still be useful to use Tox to make sure you're actually testing those versions, and not whatever you happened to have in your virtualenv at the time.