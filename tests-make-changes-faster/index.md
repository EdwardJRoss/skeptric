---
categories:
- legacy code
date: '2021-06-20T06:55:20+10:00'
image: /images/first_tests.jpg
title: Making Changes Faster with Tests
---

I used to think the whole point of software verifications like types and tests was to ensure a piece of software worked as specified.
Consequently if a piece of software already worked there wasn't much point in adding automated tests; sure we might find a few edge cases that didn't work, but we already would have had the ones that impacted end users in bug reports.
I now think the primary benefit of verifications is about making software easier to change without losing quality by introducing regressions.
Tests can directly help prevent regressions by verifying certain edge cases, but they can also increase confidence that the software is working correctly allowing less careful changes, and encourage a structure of modular, independently tested code which is easier to understand and change.

## Preventing regressions

The first time I came across the idea of regression tests was in my first professional job ad Haese Mathematics, when I was writing code that simplified algebraic expressions.
A colleague, Troy Cruickshank, had written a rules engine that recursively applied simplifications until the expression didn't change.
So for example we would have a list of rules like `1 * x -> x` and `a*x + b*x -> (a+b)*x`.
However as we grew this list of rules it became harder to know what was going to happen.
The order of the rules was very important, since some rules required the expression to be in a certain form, that other rules may help or hinder.
We'd suddenly find in manual testing that certain expressions would come out in a non-simplified form.
For example maybe we had the rules above and found the expression `1 * x + 2*x -> x + 2*x` because we applied `1*x -> x` first, when we really wanted `3*x`.
This meant I would spend a *long* time thinking before adding any rule, trying to think through any of these sorts of interactions, and I'd still miss them.

My manager, Adrian Blackburn, suggested that I right some unit tests to prevent these regressions.
The idea of coming up with test cases seemed overwhelming - how do I choose what to test?
I tested `1 + 1 -> 2`, `x + x -> 2*x`, and dozens of other cases, but I wasn't really sure they were adding any value.
However what did add value was adding real regressions making sure we didn't make the same mistake twice.

I never got fully comfortable with that rule list, but the regression tests did save time, because when I came back to it a month later and made a change the tests would immediately tell me if I'd made the same mistake again, instead of spending hours looking for and reproducing it.

## Making code easier to change

When I went to the [Compose Conference](http://www.composeconference.org/) I was talking to an advocate of strongly typed functional languages (think Haskell).
I asked them why they liked type systems so much, and they said it means when they refactor their code they know if they made a mistake immediately.
This was a revelation to me - static typing was a tool to help them change the code.
This isn't a new idea, in [Martin Fowler's *Refactoring*](https://martinfowler.com/books/refactoring.html) he talks about how unit tests let you make more aggressive refactorings in small steps, and get quick feedback if you made a mistake.

I've seen this first hand in data science code; the code had no tests, were mainly SQL, and the only way I could verify I hadn't broken the code was to run it before and after the change and run a [diff test](/diff-tests).
But this step took *hours* to run, and so I wouldn't wast a whole run on a small change, I'd try to make multiple changes.
This meant if I broke something I'd have to spend a fair bit of time debugging what I broke.
I could reduce the time taken by sampling the input dataset (although this may mean I'd miss some edge cases), and optimising the code, but it was still slow.

Over time my team refactored this code into small functions in Python that processed the data and could be independently tested (in this case the data was small enough to fit in memory).
We could run unit tests in seconds that validated or invalidated a change.
They also were useful documentation for on-boarding people to the project about what the code actually did.
It meant we went from taking months to make a substantial change being able to break it into small pieces that took days.
While big, slow integration tests have value, it's the quick small unit tests that really help make faster changes.

## Overfitting Tests

Tests that make the code harder to change are bad tests.
If you're testing a lot of things together, spending a long time setting up, and testing the details of the implementation then the test is likely overfit to the code and hard to change.

I once wanted to make changes to an in-house scheduling system that allowed us to prioritise tasks.
Occasionally it would take hours to run an ETL job, and we needed to put the time sensitive tasks to the top.
The change was relatively straightforward, but updating the tests took as long as writing the feature.

The reason was the tests I had to change were many dozens of lines long.
It effectively mocked another in-house system in painstaking detail, created a bunch of data and tested exactly what the output would be like for a particular flow of events.
I then had to propagate the prioritisation field through the system and work through how it changed the set of interactions at each step in the process.
I eventually made it all work, but it was painful and took a long time to get reviewed.


## Better code

Good tests lead to better code.
The easiest tests to write are on small, pure functions that do one thing.
Tests on these functions are clear and demonstrative.
The easiest functions to understand are small, pure functions.
Side effects and hence mocking in tests, can never be fully avoided.
But they can be moved to the edges and the majority of code can be pure functions transforming data.

I used to think the main benefit of modularity in software was to have components you reused in lots of places.
I've found in practice it's hard to reuse a lot of code, as it's very specific to a problem.
However having a system of small components that can be easily verified to work correctly individually *is* a large benefit.
Then things like types can help verify you've put them together correctly and you can quickly change code and be pretty confident it will run correctly.


A common problem I've seen in data science code is system calls to get the current date, with the underlying assumption that the date won't change (which can be false in a long running process, leading to strange bugs).
Making the date an explicit parameter makes it clearer where dates are being used, and forces someone reading the code to think about how the dates flow through the code.
It also happens to make the code easier to test, and to understand.