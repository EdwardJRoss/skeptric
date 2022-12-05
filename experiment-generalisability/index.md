---
categories:
- statistics
- data
date: '2020-11-01T21:03:08+11:00'
image: /images/ab-test.jpg
title: Experimental Generalisability
---

Experiments reveal the relationship between inputs and outcomes.
With statistical methods you can often, with enough observations, tell whether there's a strong relationship or if it's just noise.
However it's much harder to know how generally the relationship holds, but it's essential for making decisions.

Suppose you're testing two alternate designs for a website.
One has a red and green button with a santa hat and bauble, and the other has a blue button.
You run an A/B test over Christmas and the first design has a significantly better conversion rate (p < 0.001).
Should you have a button with a santa hat and bauble all year long?

The problem is that this condition won't generalise.
You ran the experiment at a certain time of year (Christmas) where the relation could be impacted by that time of year.
Any relation you find may not hold at any other time.

This over-generalisation occurs in social science studies all the time.
They will run some specific experiment on a group of undergraduates at their own university and then claim that it is sufficient evidence for a broad theory.
But really they've just shown that the particular subgroup they've chosen has acted in that way.

There's a misconception that null hypothesis testing means you don't make any (or many) assumptions about your experiment.
But when you use the results you need some theory about how it generalises to other samples in different places and conditions.
And for this you need some theory of mechanism of how it works, or what factors could contribute to it.