---
categories:
- data
date: '2020-07-15T20:02:33+10:00'
image: /images/data-model.png
title: Data Models
---

Information is useful in that it helps [make better decisions](/analysis-decision).
This is much easier if the data is represented in a way that closely match the conceptual model of the business.
Building a useful view of the data can dramatically decrease the time and cost of answering questions and even elevate the conversation to answering deeper questions about the business.

A typical example of where analysis can help is trying to increase revenue of a digitally sold product.
Maybe you could increase marketing activity to attract new customers - but which acquisition channels should you focus on?
Or you could make it [easier](/power-of-easy) to make a purchase - but where are potential customers getting stuck?
Maybe you could cross-sell existing customers, but what products should you recommend to a customer?
A good answer to these questions can contribute significantly to revenue, but they require good behavioural information.

Digital products make it really easy to gather data.
You can just check your server logs to see the activity of your website.
If you keep these logs, and have a way of tracking acquisition channels, you've got all the data you need to answer these questions.

However it's really hard to go from logs of events to answer these kinds of questions because the data model doesn't match your product.
In all of the examples above there's some notion of conversion opportunity; when a prospective customer comes to make a purchase do they make one?
You can almost certainly identify a purchase because it will trigger some sort of delivery process, but how can you identify a prospective customer and tie it to the purchase?
You can come up with a rule to identify individuals, and link it to the purchase event.
But you're really interested in acquisition channels, you'll remove individuals that have been to your website before.
And you'll need to find how they arrived at the site, identifying the first page.

This is all possible, but there's a lot of implementation work to create each piece.
But there are lots of questions to ask and many of them have these similar concepts.
You'll end up writing the same things over and over, especially if you're using SQL which doesn't have language abstractions.
Over time you'll spend a lot of time rebuilding the same abstractions from the event data.

However you could build a view of the data that fits these questions much more closely, the [Google Analytics Schema](https://support.google.com/analytics/answer/3437719) is a good example of this.
It's got a notion of a visitorID to identify an individual and a session or "visit" which represents a browsing session.
Each session has a trafficSource identifying how the visit was acquired, and eCommerceActions which summarise the purchase behaviour.
It's even got a "newVisits" field to distinguish new versus returning visits.
A lot of the questions we asked before are now a very simple query because the data is presented conceptually at a level closer to the questions.

When questions are easier to answer that enables asking deeper questions.
We can start digging into the conversion funnel, identifying key events on the path to purchase to better understand the friction.
Once the low-lying fruit is gone we can start [segmenting customers](/clustering-segmentation) based on their behaviour to understand how we can better serve specific groups of customers.
We can also start [understanding how products relate](/recommendation-graph) to better drive cross-selling.
However again asking questions about funnels, identifying groups of behaviour or digging up related products starts becoming tedious.
We can materialise these concepts and then lift the views to incorporate these concepts to make it easier to answer these questions, and start asking deeper questions again.

It's definitely important to keep the event level data separately from these views.
You can always rebuild the concepts, but it's not always easy to extract the raw event data.
The views by their nature implement business rules, and as the product changes some of the concepts will become outdated.
It's really easy to get stuck with a bad data model, but if you keep the raw data and the methodology of building views then it's at least possible to build a new version and phase out the old one.

There's obviously a cost to building new concepts and views, and doing it too much can leave a mess of data (instead of a mess of code).
However in my experience the problem is that people are unwilling (or unable) to create new views, more than that they are creating too many.
There is an exception in the intermediate tables someone created but no one knows how it was made; which is why the transform logic needs to be stored as a version controlled asset.
Now that data storage and processing is much cheaper this should become the norm rather than the exception.