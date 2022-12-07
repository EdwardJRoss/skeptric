---
categories:
- programming
- blog
date: '2020-06-20T08:00:00+10:00'
image: /images/cron.png
title: Scheduling Github Actions
---

I use [Github actions](/github-actions) to publish [daily articles](/50-daily-articles) via Hugo.
I had set it up to publish on push, but sometimes I future date articles to have a backlog.
This means that they won't be published until my next commit or manual publish action.
To fix this I've set up a [scheduled action](https://help.github.com/en/actions/reference/events-that-trigger-workflows#scheduled-events-schedule) to run just after 8am in UTC+10 (close to my timezone in Melbourne, Australia) every day.

By default Hugo will not publish articles with a future date, so it's easy to keep a backlog by setting the `date` in [front matter](https://gohugo.io/content-management/front-matter/) to a future date.
By convention I tend to set it to 8am in UTC+10 (e.g. `date: "2020-06-20T08:00:00+10:00"`) for that day.
I can still preview these posts locally by passing the `--buildFuture` flag to `hugo serve`.

So if I want these articles to automatically be published I need to run the action just after this time.
The time 08:00 in UTC+10 is 22:00 in UTC (on the previous day), so I want it to run daily at, say, 22:04.
Github scheduled actions use a crontab syntax, so I updated the configuration to run at [4 22 * * *](https://crontab.guru/#4_22_*_*_*).

```yaml
on:
  push:
    branches: [ master ]
  schedule:
    - cron: '4 22 * * *'
```

That's all there is to it.
Of course it would be straightforward to set this up on a Raspberry Pi with cron and a git server with a [post-receive hook](https://www.digitalocean.com/community/tutorials/how-to-use-git-hooks-to-automate-development-and-deployment-tasks).
But it's convenient to use managed services to not have to worry about maintaining, and especially securing, the server.