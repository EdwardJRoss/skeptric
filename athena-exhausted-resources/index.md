---
categories:
- athena
- presto
date: '2020-11-10T18:40:02+11:00'
image: /images/athena_query_waiting.png
title: Running out of Resources on AWS Athena
---

AWS Athena is a managed version of Presto, a distributed database.
It's very convenient to be able to run SQL queries on large datasets, such as [Common Crawl's Index](/common-crawl-index-athena), without having to deal with managing the infrastructure of big data.
However the downside of a managed service is when you hit its limits there's no way of increasing resources.

Today I was running some queries for a regular reporting pipeline in Athena when I got failure with the error `Query exhausted resources at this scale factor.`
The query was running out of memory, but I had no idea why.
I had run this query before with no issues.
I reran the pipeline and then it failed with the same error at a different step.

The problem is that there is no visibility on why things are failing, and no levers to get more resources.
There's just enough differences between Athena and Presto that if I spun up my own Presto cluster, which I could scale to any size, I'd have to make some small changes to my queries to have them run successfully.
There was a good risk that the process was broken for a couple of days.

I kept on retrying and eventually it reran.
I talked to someone else who had similar problems, and it sounds like it may have been an issue on the AWS end.
But I'll never really know and this is the risk.

A managed service with no levers like Athena, or Google BigQuery, is extremely convenient to run data pipelines with.
But the problem is that if your data grows or the service changes your pipeline might hit the limits and you may have to interrupt your service and either rewrite your pipeline or migrate to another service.
It's worth considering this risk and it may be worth investing in a solution that allows you to scale up the infrastructure such as Spark.