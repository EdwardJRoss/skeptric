---
categories:
- data
date: '2020-08-11T21:00:00+10:00'
image: /images/search_session.jpg
title: Sessionisation Experiments
---

You don't need a lot of data to prove a point.
People often think statistics requires big expensive datasets that cost a lot to acquire.
However in relatively unexplored spaces a small amount of data can have high yield in [changing a decision](/analysis-decision).

I've been working on some problems around web sessionisation.
The underlying model is that when someone visits your website they may come at different times for different reasons.
A session (sometimes called a "visit") tries to capture this intent.
The standard implementation in web analytics tools like Google Analytics and Adobe Analytics is a sequence of page views with no more than a 30 minute gap between them (with some differences in edge cases).

I was debating with some members of my team around what the best period of sessionisation was in our usecase, and how good it needed to be.
There were different viewpoints based on people's intuitions around how people behave.
In the end a few simple experiments resolved the issues.

The easiest test is invariance; if we change our definitions and it has negligible impact on the output, then it doesn't matter which we use.
An example of this is Google Analytics "end of day" rule; sessions will break at the end of the day in a specified timezone (which stops infinitely long sessions, and makes some forms of reporting easier).
Running an alternative model without the "end of day" rule had a similar number of sessions, which showed this had minor impact on outcomes.

It's harder when there's a real difference because you need a ground truth.
However you don't need a lot of data.
In this case there were fairly clear types of intent we were trying to distinguish, which could be annotated by hand from the weblogs.
While not a great job it would be under an hour's work for a user week in most cases.

We picked 5 random user-weeks and annotated them in around a day.
We could measure the accuracy in separating intent in two scenarios.
By the [rule of 5](/rule-of-five) a 95% confidence interval for the median user-week accuracy is between the lowest and the highest.
In this case one scenario was clearly better in these cases which gave confidence it would work better for typical users.

However there was still concern on how it would treat edge cases.
In this case we used the opposite of invariance: we isolated the cases where there was the most difference between the two models.
We could then annotate them and again found the same rule worked better in most cases.

This kind of approach won't detect small differences, but we didn't need it to.
All we needed was enough information to make a decision, and to get an idea of how much was left to gain from further improvements.
This was an effective way of ending weeks of debate in a few days.

It's surprising how often these kinds of simple data collection methods for decision making are overlooked.