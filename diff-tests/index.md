---
categories:
- legacy code
date: '2020-09-17T22:33:13+10:00'
image: /images/spot_difference.png
title: Diff Tests
---

When making changes to code tests are a great way to make sure you haven't inadvertently introduced regressions.
This means that you can make changes much faster with more confidence, knowing that your tests will catch many careless mistakes.
But what do you do when you're working with a legacy codebase that doesn't have any tests?
One method is creating diff tests; testing how your changes impact the output.

For batch model training or ETL pipeline there's typically a natural way to do this.
You can take a sample of data and run it through the pipeline and inspect the output.
If you run the data pipeline twice you should get (roughly) the same result.

The basic process is to create a test harness that can tell if two outputs are the same.
For a file this can be as simple as running `diff` across the outputs.
Sometimes there's state you need to remove from the output or the tests; for example `gzip` contains a timestamp but `zdiff` ignores this in the calculations.

Another issue is if the output is non-deterministic due to randomness or race conditions.
Sometimes this can be configured away by setting random seeds and running on a single thread on a single machine; but sometimes not.
When not then you'll have to run stochastic tests that check whether things are approximately the same; this is more difficult to implement and more difficult to determine bounds (you can do things like run the existing pipeline many times to get some bounds).

Wherever possible you should freeze inputs by using immutable snapshots from a database, letting time be a configuration parameter (rather than reading the system clock directly throughout the code), and caching any resources fetched from external APIs.
These typically make the system easier to test, to inspect, recover from errors and run in small pieces so are generally beneficial.

It's important for this workflow to get the code running as quickly as possible to allow fast iteration of changes.
One way to do this is to [isolate parts of the dataflow](/dataflow-chasing) to the part you're modifying and serialise the output somewhere; this reduces the amount of processing you need to run.
Another is take a smaller sample of the data (which can be a bottleneck); however the smaller the sample the less edge cases you're likely to find (it's probably good to run a larger sample through once a day).
Finally there are often ways to speed up the bottlenecks like paralellising (using [multiprocessing](/multiprocess-download) or [futures](/multiprocesing-future) in Python), through caching frequent costly operations, or optimising the code for example by vectorising.

This methodology doesn't replace tests.
Tests can much more efficient at verifying the code works on a range of specified cases; and lets you specify all the edge cases that have given you trouble before (which may not occur in your sample dataset).
Although the idea has some resonance with [property based testing](/property-based-testing) where you specify a relation; in this case that the new implementation should give the same result as the old implementation (within some bounds).
However it lets you work on a codebase with no tests, and allows you get it into a workable state when you can start implementing tests.