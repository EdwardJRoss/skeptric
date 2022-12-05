---
categories:
- maths
date: '2020-07-10T08:00:00+10:00'
image: /images/mvt.svg
title: Mean Value Theorem
---

I remember three things from lectures in my first year of university.
One is a chemistry professor acting out the three vibrational modes of water; his head being the Oxygen atom and his hands being the Hydrogen atoms.
Another is [Rod Crewther](https://en.wikipedia.org/wiki/Rod_Crewther) demonstrating torque by showing how difficult it to open a heavy lecture door by pushing at the hinge.
The third is how [Nick Buchdahl](http://www.maths.adelaide.edu.au/people/nicholas.buchdahl) illustrated the mean value theorem.

After a big night out at a pub you somehow managed to get home to your bed.
You're not quite sure how you got home; there's a vague recollection of wandering home, from lampost to lampost (but bouncing off them smoothly, so your path stays differentiable).
Then the mean value theorem says at some point you were going in the right direction.

This is a pretty good way of capturing the mean value theorem.
The simple form we learned is that if a function $$f:[a,b]\to\mathbb{R}$$ is differentiable on its interior and continuous at its endpoints, then there is a point c in the interval such that $$f'(c)=\frac{f(b)-f(a)}{b-a}$$.
The right hand side is the "right direction"; the slope of the straight line path from the pub to your bed.
The left hand side is the direction you're going at the instant c; the tangent to the path.

For the analogy to work you have to be able to model your path as a function of one variable; in particular this means you can never cross your own path, otherwise there's no way to make an x-axis.
This wouldn't be true for many drunkards walks, and in higher dimensions the generalisation isn't as straightforward.

But this is beside the point; the story was compelling enough to make me remember the mean value theorem years later and ask questions on whether it applies to higher dimensions.
Similarly the vibrational modes of water and understanding of torque have stuck with me.
These three excellent teachers by demonstrating their point with a very visual story (and being willing to look a bit foolish) taught me something that I still remember.
If you ever want something to stay with people make sure you thread it into a story they can visualise.