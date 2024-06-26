---
categories:
- data
date: '2020-05-01T21:02:06+10:00'
image: /images/4am.png
title: 4am Rule for timeseries
---

When you've got a timeseries that doesn't have a timezone attched to it the natural question is "what timezone is this data from?"
Sometimes it's UTC, sometimes it's the timezone of the server, otherwise it could be the timezone of one of the locations it's about (and it may or may not change with daylight savings).
When it's people's web activity there's a simple heuristic to check this: the activity will be minimum between 3am and 5am.
I've found this 4am rule to be pretty consistent and useful for any timezone difference greater than an hour.

If you've got people from multiple timezones it's useful to check accross the people to find out whether it's a localised time.
There's often a way to infer gross location, for example through an IP lookup or by looking at content or activity specific to particular areas.

For example the Australian Broadcast Corporation [released hourly data of web activity from 2016](https://data.gov.au/dataset/ds-dga-316060ae-e49d-4e39-949a-ed3fdaede18d/details).
However in the data description it doesn't mention what time zone the data is in.
A quick plot of the web pageview volume for 1 day in May shows the trough is at 3am and it likely corresponds to the main timezone for Australia: Australian Eastern Standard Time.

![Pageviews by Hour is minimum at 3-4am](/images/4am.png)

Western Australia is 2 hours behind the eastern states of Australia, and it's not clear whether it would be measured in local time or AEST.
The pages have metadata attached which includes a field of pipe separated content topics, which often includes a state or city.
It seems likely information about a state or city will be most viewed by people in that location.
Looking at a plot of a week of web pageviews by hour based on the location of the content shows the trough in Western Australia is also around 3-5am.
This means it's likely to be *local* time and the timestamps are probably based on the users device.

![Pageviews by Hour by State is minimum at 3-4am](/images/4am_state.png)

The metadata also has a "content first published" field, and it's not clear what timezone that is in.
However if the page view data is in local time it will look like views from WA are lagged 2 hours relative to publishing time relative to other states.
This is rather difficult to verify; it would be easier if it led by 2 hours because then we may see page views before it's published.
In this case because the population of WA is relatively small it can be ignored for most aggregate statistics.

The 4am rule is useful for verifying what timezone a digital activity dataset is in to within 2 hours.
It's simple enough to check quickly with a database query when there's no reliable information on the timezone.
Whenever publishing data use [ISO-8601 format with a timezone](https://en.wikipedia.org/wiki/ISO_8601) so the timestamp data is unambiguous and consumers don't need to use this rule.