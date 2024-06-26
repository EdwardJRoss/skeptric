---
categories:
- ''
date: '2020-03-25T21:00:59+11:00'
draft: true
image: ''
title: Dynamic Rollup tables in Bigquery with Hyperloglog
---

```SQL
SELECT device.operatingSystem,
       PARSE_DATE('%Y%m%d', _TABLE_SUFFIX) as date,
       count(fullVisitorId) as users, 
       sum(totals.pageviews) as views, 
       sum(totals.totalTransactionRevenue) as revenue
FROM `bigquery-public-data.google_analytics_sample.ga_sessions_*` 
WHERE _TABLE_SUFFIX BETWEEN '20160801' and '20160803'
GROUP BY 1, 2
ORDER BY date, views desc
```

Citus article
https://www.citusdata.com/blog/2017/06/30/efficient-rollup-with-hyperloglog-on-postgres/Hll


Export schema
https://support.google.com/analytics/answer/3437719

Dataset:
https://support.google.com/analytics/answer/7586738

Wildcard: https://cloud.google.com/bigquery/docs/querying-wildcard-tables