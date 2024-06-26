---
categories:
- general
date: '2020-06-17T11:59:15+10:00'
image: /images/checking.jpg
title: Checking your Work
---

One of the most important abilities of an analyst is to be able to check your work.
It's really easy to get incorrect data, have issues in data processing, or even misunderstand what the output means.
But if your work is valuable enough to [change a decision](/analysis-decision) it's worth doing whatever you can to check it's right.

When you get to the end of a long analysis it seems like a time to relax and be glad the hard work is over.
But the [fourth step of How to Solve It](https://en.wikipedia.org/wiki/How_to_Solve_It) is to "Review and Extend".
There are typically lots of ways you can check your result makes sense.
Is there another way you can calculate it?
How does it fit with other things you know about?

This is why it can be very handy to keep a bunch of basic numbers about your domain at hand.
How many users does your website get each week?
How many emails do you send out, what percentage of them are opened?
These often allow you to do a quick [Fermi Estimate](https://en.wikipedia.org/wiki/Fermi_problem) to check an answer is in the right ballpark, by relating it to one of these other things.

Thinking through how it to relates to other things can also help you gain insight.
If the pageviews went up but weekly users went down, are you getting more churn in your infrequent users?
What would this mean about your conversions?
What sources of traffic are they asssociated with?

Ultimately the best way to have reliable output is to monitor your input and transform with well small and well tested steps; tracking invariants with [property based testing](/property-based-testing).
But this kind of system can be very slow to build and you can lose the intuition of how the parts relate to the whole.
Checking a result by relating it to other known quantities is a very effective technique (and the only reason people can use an unstructured tool like [Excel for analyisis](/using-excel)).