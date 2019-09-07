---
title: "Property Based Testing - A thousand test cases in a single line"
date: 2019-07-02T21:44:01+10:00
draft: true
---

Property based testing lets you specify rules that a function being tested will satisfy over a wide range of inputs.
This specifies how to throughly test a function without coming up with a detailed set of test cases.

For example instead of writing a specific test case like `sort([1, 3, 2]) == [1, 2, 3]`, you could state that the input and output of sort should contain exactly the same elements for any valid input.

## Specifying more complete tests with less code

Writing simple, disjoint test cases that each test exactly one thing and cover most likely failure modes is difficult.
Often there are simple properties that can succinctly replace a large number of tests.
For example a banking application may have a complex distributed implementation, and it's hard to test all the possible failure modes.
However a simple property is that the amount in an account should balance against the deposits and withdrawals.
This declarative property could be tested against a multitude of possible concurrent transactions to ensure the code is working correctly.

Some common types of properties include:

* Applying the function twice gives the same result as applying it once (idempotency). This test on unicode canonicalisation would help prevent an [account hijacking vulnerability at Spotify](https://labs.spotify.com/2013/06/18/creative-usernames/).
* It gives the same result as a simple brute-force solution. For example finding the shortest path in a graph by enumeration.
* The output takes a specific form; like the output of the sort function is in non-descending order.

Properties are complementary to individually specified test cases.
Cases can be clearer and make better documentation, and some functions don't have properties that are easy to check and cover everything you want to test.
However when properties exist they can be used to test on automatically generated data more complex than you would write in a test case.


## Generating test cases

Once we've specificed some properties we need some input data to test whether the function satisfies these properties.
There are a few choices for test data:

* Random data from a specified input distribution. This is how Haskell's [QuickCheck](http://hackage.haskell.org/package/QuickCheck) works
* Enumerating over all cases from simplest to most complex - for example starting with an empty list and trying longer lists. This is how Haskell's [SmallCheck](http://hackage.haskell.org/package/smallcheck) works and is good for expensive properties.
* Using data samples from production workloads.
* Getting examples of likely edge cases (for example rare unicode characters, extreme floating point numbers).

## Example of Property Based Testing

I recently needed to find the minimum number of padding elements I would need to add to a list to divide it into sublists of a pre-defined lenth.

To do this I wanted a helper function of signature in pseudo-python of: `pad_divide(numerator: int > 0, denominator: int > 0) ->  (pad, div): Tuple[int, int]`

By definition `pad_divide` must satisfy the properties:

* `pad` and `div` are integers
* `numerator + pad == denominator * div`
* `0 <= pad < denominator` (since if `pad >= denominator` then `pad - denominator` would also be an acceptable padding length).

In fact these properties uniquely define the function (we could inefficently solve it by searching through all pad sizes).
Writing these tests up front, and testing them on all numerators and denominators between 1 and 30 helped me get to the right solution more quickly.

## Implementing Property Based Testing

For simple cases (like above) it's possible to implement property based testing by hand; but if you want to use it extensively (and do clever things like find the simplest failing case) there are libraries for many languages:

* This type of testing originated in Haskell with [QuickCheck](http://hackage.haskell.org/package/QuickCheck) and later [SmallCheck](http://hackage.haskell.org/package/smallcheck).
* Python has [Hypothesis](https://github.com/HypothesisWorks/hypothesis)
* Scala has [ScalaCheck](https://www.scalacheck.org/)
* JavaScript has [jsverify](http://jsverify.github.io/)

To understand more of how these are implemented the original papers for [QuickCheck](https://www.cs.tufts.edu/~nr/cs257/archive/john-hughes/quick.pdf) and [SmallCheck](https://www.cs.york.ac.uk/fp/smallcheck/smallcheck.pdf) are illuminating.
