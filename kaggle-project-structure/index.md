---
categories:
- data
- programming
date: '2020-11-27T18:08:30+11:00'
image: /images/kaggle_hamburger.png
title: Structuring a Project Like a Kaggle Competition
---

Analytics projects are messy.
It's rarely clear at the start how to frame the business problem, whether a given approach will actually work, and if you can get it adopted by your partners.
However once you have a framing the modelling part can be iterated on quickly by structuring the project like a Kaggle Competition.

The modelling part of analytics projects will go smoothly only if you have clear evaluation criteria.
There are methodologies like [CRISP-DM](https://www.sv-europe.com/crisp-dm-methodology) or [Jeremy Howard's Data Project Checklist](https://www.fast.ai/2020/01/07/data-questionnaire/) for running a successful project.
Modelling and analytics is only one (often small) step in the process, but it's what defines it as an analytics project.
It's easy to spend a lot of time digging into the data and trying different approaches, but if you don't have a clear evaluation criteria it's really hard to compare them.

An imperfect evaluation criteria is often better than none.
Your evaluation criteria should align with the business goals, but will often have to be a proxy for the real thing.
Because of this getting the absolutely best score is often *not* worthwhile; there will be diminishing returns as models get more complex.
But it gives you a guidepost and helps identify whether something is working at all.
Ideally you have [a simple evaluation metric](/simple-metrics) that you can communicate back to stakeholders to demonstrate you're actually doing something.
Try to come up with just one primary metric, with at most a couple of guardrail metrics (i.e. constraints), otherwise you can spend a lot of time deciding between tradeoffs.

Once you've got an evaluation criteria, and created a good training/development/test set data split, then it's easy to compare solutions.
You should start with a [very simple model](/simple-models) as a baseline and then build on that.
While it can be fun trying to build bigger better models, if you're developing a new analytics product or service try to get to "good enough".
Often picking a small subset of the possible data (say a particular region or time period) will make it much faster to prove an approach.
If you can prove it's viable in the field then you can come back and expand and improve it.

It's easy to get stuck with a mediocre implementation if you don't set a clear structure.
If you start with a solution you'll start building your evaluation around the structure of your solution.
This makes it harder to test alternative solutions later on, and the evaluation may even get stuck to implementation details.

When the solution is in production you can then evaluate your metrics online and reconcile them with your offline test metrics (which may be different if you inadvertently had a data leak!)
Then it can be refined and optimised over time as it becomes more obvious what's important in production.
But you can keep the same project structure for future improvement cycles.

This technique won't work for all problems, and it's not always easy to come up with an offline evaluation metric and set.
But when you can using this structure will make your modelling iterate faster and set up success for monitoring and improvement if the solution gets adopted.