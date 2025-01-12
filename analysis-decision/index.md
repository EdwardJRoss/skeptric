---
categories:
- data
date: '2020-04-22T22:40:22+10:00'
image: /images/decision.png
title: Analysis Needs to Change A Decision
---

Any analysis where the results won't change a decision is worthless.
Before even thinking of getting any data it's worth being clear on how it impacts the decision.

There's lots of reasons people want an analysis.
Sometimes it's to confirm what they already believe (and they'll discount anything that tells them otherwise).
Sometimes it's to prove to others something they believe; possibly to inform a decision someone else is making.
But it's most valuable when it effects a decision they can make with an outcome they care about.

I always find it useful to clarify this and do some scenario modelling before even planning out an analysis.
Some useful questions to ask are:

* What's the problem you're trying to solve?
* What does a good outcome look like?
* What outcome do you expect? How sure are you?
* What would you do if the result looked like this?
* How big is the impact of getting it wrong?

When you understand this you can be clear how rigorous you need to be in the analysis.

For example suppose that you want to know whether a more personalised subject line will improve click through rates.
There's a lot of industry knowledge suggesting that's the case, but it's not really clear how much it will improve things.
You could run an A/B test to see whether it improves click through rates by more than 2 percentage points, say.
If it makes the click through rates worse you would definitely stick to your current version.
But if it *doesn't* make any difference (null case) would you stick to your current version?

Industry best practice says that you should personalise your subject line, and you're pretty happy with what you've got.
You may as well stick with the new version, it's not making things any worse.

How bad is it if you're wrong?
Maybe you're going to be doing a bunch of experimentation on the subject line so it doesn't matter too much and you set your confidence thresholds lower.

Standard statistical analysis often focuses more on *confidence* that a result is different than 0, than how significant the difference is. 
It also doesn't focus on how impactful the difference is on downstream decisions, but this is crucial in understanding what analysis to do.