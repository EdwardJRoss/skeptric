---
categories:
- maths
date: '2020-08-04T08:00:00+10:00'
image: /images/probability_space.svg
title: Symmetry in probability
---

The simplest way to model probability of a system is through symmetry.
For example the concept of a "fair" coin means there are two possible outcomes that are indistinguishable.
Because each result is equally likely the outcome is 50/50 heads or tails.

Similarly for a fair die there are 6 possible outcomes, that are all equally likely.
This means they each have the probability 1/6.

The idea of symmetry is behind random sampling.
If we want to understand a population we can take a number of random cases and it tells us something about the whole.
However this is only true if the sample is random with respect to the properties we're measuring.
That is if we exchanged people randomly we would be equally likely to measure them.

Another example is a spinner, like a roulette wheel.
The model is that a fair spin is equally likely to land anywhere on the circumference circle.
So by symmetry the probability of an outcome is proportional to the length of the arc it subtends on the circle.
This also happens to be proportional to the angle of the arc, which is proportional to the area of the arc.

Another more elaborate example would be a sphere with labelled regions that can spin on both axes, and a pin pointing to a point on the sphere (say the topmost).
Then if it spins freely in any direction the probability of landing on any point is proportional to the surface area on the sphere.

You could cut the sphere in half and flatten it into two circles.
This allows you to make the same reasoning about a dartboard; the probability of an uncontrolled throw landing in a region is proportional to its area.
This kind of geometric reasoning is behind [probability squares](/probability-square).

This idea can be generalised to any sort of symmetry.
In practice all we really need is the finite examples like samples, dice and coins since real populations are finite.
However the more general case in mathematics is an extension of the Haar measure theorem.
It's something like any free and transitive continuous group action on a compact Hausdorff space induces a unique probability measure.
For the finite cases it's the symmetric group, for the circle U(1) and for the sphere SO(3) (which isn't covered by the Haar measure theorem since a sphere is not a group).
But none of this will actually help you with probability and statistics, because again real populations are finite (we just sometimes pretend they're infinite to make the maths easier).
And even if you do need it, the geometric intuition about areas is likely to help you more than group theory.

In practice all the symmetry ideas break down.
We have unfair outcomes and so we have to introduce a bias parameter.
It's really hard to get a truly random survey sample and so we have to try to model the bias.

However the ideas are still useful for simplifying hard problems.
In survey sampling we often assume that a measurement is symmetric within some demographic attributes and just try to weight the survey based on its demographic composition.
This is an old tradition, but can [go terribly wrong](https://www.nytimes.com/2016/10/13/upshot/how-one-19-year-old-illinois-man-is-distorting-national-polling-averages.html).
Unfortunately for a very biased sample there are limits on the amount of population information it's possible to extract.