---
categories:
- data
date: '2020-08-31T21:00:00+10:00'
image: /images/forecasting.gif
title: From Descriptive to Predictive Analytics
---

The starting point for an analysis is often summary statistics, such as the mean or the median.
For some of these you're going to want it more precisely, more timely or cut by thinner segments.
When the data gets too volatile to report on it's a good time to re-frame the descriptive statistics as a predictive problem.

Businesses often have a lot of reporting around important metrics cut by key segments.
Monthly average revenue per customer by industry.
Weekly average sales per salesperson by territory.
Percentage of daily website visits that bounce by traffic source.

Some of these may have a large impact on decisions, but the data is quite sparse and the imprecision means the business can't make a timely decision.
Where do we need to recruit more salespeople?
What stock do we need to increase the supply of?
Should we keep spending on our marketing campaign to drive consideration?

You can re-frame these descriptive statistics as predictive problems.
The mean and median are the constants that minimise mean squared error and mean absolute error respectively.
For example the weekly average sales per salesperson by territory could be considered the model: `sales ~ week x territory`.
That is we treat each week and territory as independent and fit a [constant model](/constant-models) to each.

But you know that the weeks aren't independent.
A traditional approach to this is to apply a time series forecasting method.
Another approach is to model over relevant factors related to each week; such as sales and budget cycles that have a large impact on sales.

You also know the sales territories aren't independent.
Some of the sales territories are very close together, and will have similar characteristics.
Maybe you know some characteristics of territories, such as the types and density of customers they contain that impact sales.

You could combine all this to create a more accurate model of sales.
Your previous segmentation is a good baseline, and by testing on a holdout set of the latest week you could determine whether your model is actually better at predicting sales.
This could mean a better understanding of the potential return on investment of putting more salespeople in certain territories, or allow better incentive setting.

There is a danger here that the models will sometimes be wrong.
Especially in turbulent times, very different to the history they've been trained on, they will confidently mispredict.
They're also harder to interpret and explain; they require building trust with stakeholders.

But if the models will inform decisions that lead to significantly better outcomes they may be worth the risk and complexity.