---
categories:
- data
date: '2020-06-01T19:53:31+10:00'
image: /images/line_chart.png
title: Simple Metrics
---

I have a tendency to create really complex metrics. 
Sometimes when I'm analysing data I'll need to transform the data to understand it.
I often calculate the ratio of common metrics to get a more stable rate.
Or when building a machine learning model I'll find that log-loss or root mean square log error is the right metric.
This can be appropriate for gaining insight or training a model, but it's not good for communication.

As an analyst I spend many hours thinking about quantities and it's relatively easy for me to understand a new quantity.
And when building a model or analysis I spend more time thinking about the metrics I'm using.
This can make a blind spot when communicating to someone else; I watch their eyes glaze over as I try to explain the metrics.

It's best to transform it back to something they understand.
Where possible use the standard reporting metrics they're familiar with.
An explainable metric needs to fit in a single simple sentence.
Something like "97% of the time the model predicts the value within 10% of the actual value" is likely more meaningful to stakeholders than the root mean square log error is 0.02.

Of course you should still track complex metrics internally when they're the right thing.
But for communicating value they should be as simple as possible using familiar terms.
If you really do need a new term you will need to do a lot of work to explain and establish it.