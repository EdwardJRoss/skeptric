---
date: 2020-02-07 15:18:52+11:00
draft: true
title: Working With Data Scientist Code
---

The idea of working with code written by a data scientist sends chills down the spine of many software engineers.
Most data scientists, statisticians and data analysts are extremely good at quickly manipulating data and experimenting with different approaches to a problem.
Fewer of them are skilled on writing software that is easy to maintain and deploy.
Common software development practices like automated builds, tests and modular code are rarely adhered to in data science code.
However it **is** possible to work effectively with them by producing reproducible runs, instrumenting metrics and creating diff scripts.

There's a good reason many data science and analytics codebases are hard to maintain.
Most of these projects are experimental at inception.
There is a lot of time spent understanding the data, trying different approaches, and evaluating the outcomes through different perspectives.
Often whole projects are throwaway because the data is not good enough, or the problem is too difficult to solve for the expected return.
When writing this kind of prototype code it's important to iterate quickly and have many different diagnostics of the data.
However if it's successful and the script becomes part of a regular process, then it suddenly becomes important for it to be understandable, reliable and easy to change.
When you need to work with data scientist code you should spend some of your time paying down this tech debt.

An approach I've found effective for working with data scientist code is:

- Get the code working
- Follow the data flow to divide and conquer
- Create diff scripts and reproducible runs
- Refactor the code to make it easy to change
- Make the changes

# Getting it working

> Works on my machine

Based on my experience I'm going to assume the current state is:

- You've got some collection of scripts in languages like Python, Bash, R or SQL (and maybe some SPSS or SAS)
- The scripts run in a pipeline (sometimes they're helpfully labelled in order `01_`, `02_`)
- Each script contains long chains of impenetrable imperative data munging
- It worked once, on someone's machine, but not on yours now

The first step is to use a Version Control System like Git if it's not already being used.
It's easiest just to commit all the files to start with, and sort out separating the data from the code later.

The next step is to try to get an operating environment you can run it in.

## Setting up an environment

It's much easier to work reliably with running code than with broken code.

The most common reasons code may not run are:

- Dependencies are not installed
- External resources like databases or URLs are not available
- You're running it in a different place








## Environment

Environmental assumptions: (/home/user/rperkins), datetime 2019-12-15, OSX

Restoring dependencies: Changelogs and release history, git log, bisections

Version of code (R's random number generator!)

External resources: Database changes, URLs changed content


*Strongly resist the urge to rewrite any of it from scratch.*
## Understanding it

Looking at input and output.

Start at end of function and read backwards

Refactor it

## Tackling it

Faking it: docker/VM/cloud VM

Changing things and assessing risks

Carefully checking output as expected

Divide, debug and conquer





----

An approach I've found effective for refactoring code analytics code is:

- Understand the data flow
- Divide and conquer
- Create diff scripts

- Break the code into small steps
- Create a small, representative test dataset
- Seeing the tree from the branches



# Crude tests with Diff Scripts





Common software development practices like testing, [DRY (Don't repeat yourself)](https://en.wikipedia.org/wiki/Don%27t_repeat_yourself), and automated building are uncommon in data scientist code.
However by using


For exploratory code, like a one-off analysis, this is sensible -
When the code is a one-off analysis or experimenting to see whether a problem is solvable.

Imagine that

You have to rerun that script the advanced analytics team sent over, and switch it over to the new data source.



And you have to add a new

The idea of working with code written by statisticians, analysts or data scientists puts a chill down the spine of many a software engineer.

We have to rerun that script the advanced analytics team sent to us. Oh


Data scientists, analysts, and statisticians aren't known for writing the m

How do you deal with analytics code with no tests

[Martin Fowler's Refactoring](https://refactoring.com/)


- Follow the variables

- Follow the output





# Preparing for changes

Goal: Test a change for invariance as quickly as possible

## Do you actually need to change this portion of code?

If you're not changing it, and it works, try to leave it alone


## Divide and conquer

Map out inputs -> outputs

## Gold standard technique

Static dataset and a diff script

Small with edge cases (fast)

Invariance of outputs: The diff script

## Branching and parameterisation

Remove or test branches; run multiple times to test multiple branches

Configure things to be fast; trust external dependencies

## Nondeterminism

* Randomness
* Race conditions
* External resources: time, database changes, URL content, user input

Solutions:
- Set random seeds of PRNG
- Single threading
- Mock environment
- Diff scripts that are insensitive to these changes

# Making changes

## Refactoring: Many small changes and test goldset

Small change + quick test -> quick incremental improvements

Large change + slow test -> long debugging cycles, more likely to miss errors

Occasionally run fuller test on bigger test set, more parameters, etc.
Write test cases for failures


Goals? Refactoring by Martin Fowler

Clean API, functional: Makes it easier to test
- Each function does one thing
- Functions don't mutate their inputs
- No mutable globals; few globals
- Pass time as a parameter (multiple today() problems)
- As many functions as makes it easy to read
- No dead code
- Fail fast (no warnings R!)

Add tests?

## Making the changes

Add tests
Use diff scripts to check you only changed what was intended
