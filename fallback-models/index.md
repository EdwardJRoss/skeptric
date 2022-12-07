---
categories:
- ''
date: '2021-09-14T09:02:08+10:00'
draft: true
image: /images/
title: Fallback Models
---

Suppose a marketer wants to understand the value of different acquisition channels, and calculated the average spend per acquired customer.
For large channels this is straightforward to get a stable estimate.
However if there are lots of channels with only a few customers then the estimates for those channels are likely to be very unstable, and if you rank the channels some of these are likely to be near the top.
In fact with the very limited evidence given it's likely that these channels will perform more similar to the overall average.
One way to handle this is to say if there are fewer than N observations, for some small number N, we use the average spend per acquired customer over *all channels* as our estimate.

This approach is a simple way to trade-off between bias and variance.
The overall average is the best [constant model](/constant-models); it has very low variance, but a high bias.
Estimating each group individually has a low bias, but a variance inversely proportional to the sample size (that is decreases as 1/n where n is the number of observations in that group).
So at some point as sample size increases the variance of the group estimates will contribute less than the bias of the overall average, and we should switch between the two.
The point at which this happens depends on how widely spread the groups are about their centre, and how widely spread the group centres are from each other.
A particularly interesting case is binary classification with many groups — in the opening example, this could be the probability a new acquisition converts on their first visit — where 
with only a few observations the probability is very uncertain.

This is a simple approach but can be easily extended to multiple levels of fallback.
For example perhaps our digital marketer wants to further segment the acquisition channels by marketing campaign, device and country.
For the full pivot of channel * campaign * device * country there may not be many that have enough data for a stable estimate.
However where we don't have enough data we can fall back to channel * campaign * country on the knowledge that behaviour isn't too different between devices.
Next we could fallback to channel * country on the reasoning that the differences between countries is larger than the differences between campaigns.
Then we could fallback to channel * campaign, and then to channel and finally to country then everything.
In each case when calculating the statistic for a group we include all the data that *could* fallback into that group (so, for example, the last group is all the data), but at inference an instance falls back the the most specific group for which there was information.

This approach is reasonably explainable, 



[](/categorical-embeddings)

heirarchical models