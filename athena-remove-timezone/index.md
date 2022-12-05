---
categories:
- presto
- athena
date: '2020-08-17T22:12:28+10:00'
image: /images/presto_time_data.png
title: Removing Timezone in Athena
---

When creating a table in Athena I got the error: `Invalid column type for column: Unsupported Hive type: timestamp with time zone`.
Unfortunately it can't support timestamps with timezone.
In my case all the data was in UTC so I just needed to remove the timezone to create the table.
The easiest way to do that was to cast it to a timestamp (without a timezone).

```sql
cast(event_time as timestamp)
```