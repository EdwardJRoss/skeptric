---
categories:
- sql
- data
- presto
- athena
date: '2020-08-18T23:37:33+10:00'
image: /images/sql_sessionise.png
title: Create User Sessions with SQL
---

Sometimes you may want to [experiment with sessions](/sessionisation-experiment) and need to hand-roll your own in SQL.
There's a good [mode blog on how to do this](https://mode.com/blog/finding-user-sessions-sql/).
If you're using Postgres or Greenplum you may be able to use [Apache Madlib's Sessionize](https://madlib.apache.org/docs/latest/group__grp__sessionize.html) for the basic case.
This blog post will give a very brief summary of how to do this with some examples in Presto/Athena.

The idea of a session is to capture a continuous unit of user activity.
For example if you want to report on whether a visitor to your product bought something, you need to specify the bounds of that opportunity - this is commonly a session.
The common way of doing this is to pick some "session time", typically 30 minutes, and say that if two events occur within the session time it's the same session.
There's some subtlety to this, for stable reporting you'll need some way to cap the maximum session time.
Google Analytics does this by cutting sessions at the end of a day, Adobe Analytics does this by ending a session after 12 hours.

There may also be certain kinds of events you want to consider as a new session.
You have a lot of specific knowledge of your domain, and you may have another way or better capturing what an "opportunity" to convert is.
But typically the time based approach will be the basis, and you may apply some additional rows.

The scenario is you have a user identifier, and a sequence of timestamped events (like in a [GA 360 Bigquery Export](https://support.google.com/analytics/answer/3437719?hl=en)).
You then want to group these into sessions.

| user | time                | event |
|------|---------------------|-------|
| 1    | 2020-08-18T17:00:00 | a     |
| 1    | 2020-08-18T17:15:00 | b     |
| 1    | 2020-08-18T18:00:00 | a     |
| 2    | 2020-08-18T17:45:00 | c     |

The way you do this is look at the time since the last event for each user; if this is more than 30 minutes then we start a new session.

```sql
select
  user, time, event,
  date_diff('minute',
            lag(time) over (partition by user order by time),
            time) as mins_since_last_event
from events
```

| user | time                | event | mins_since_last_event |
|------|---------------------|-------|-----------------------|
| 1    | 2020-08-18T17:00:00 | a     | NULL                  |
| 1    | 2020-08-18T17:15:00 | b     | 15                    |
| 1    | 2020-08-18T18:00:00 | a     | 45                    |
| 2    | 2020-08-18T17:45:00 | c     | NULL                  |

We can get a user specific session id by calculating the cumulative count of new session events.

```sql
select *,
  sum(case min_since_last_event is null
        or mins_since_last_event > 30
      then 1
      else 0
      end) over (partition by adv_user_id order by time) as user_session_id
from (
select
  user, time, event,
  date_diff('minute',
            lag(time) over (partition by user order by time),
            time) as mins_since_last_event
from events
)
```

| user | time                | event | mins_since_last_event | user_session_id |
|------|---------------------|-------|-----------------------|-----------------|
| 1    | 2020-08-18T17:00:00 | a     | NULL                  | 1               |
| 1    | 2020-08-18T17:15:00 | b     | 15                    | 1               |
| 1    | 2020-08-18T18:00:00 | a     | 45                    | 2               |
| 2    | 2020-08-18T17:45:00 | c     | NULL                  | 1               |

That's really all there is to it.
To get a Google Analytics style session you would also group by `date(time)`.
To get an Adobe Analytics style session requires more work; a simple way would be to do another pass over the sessions creating a new cut whenever we pass 12 hours from the session start time.

There are other ways you could customise it.
There may be certain events that effectively "reset" the clock; for example in user search sessions leaving the search context may end the session.
This could be done by starting a new session depending on the identity of the previous event.