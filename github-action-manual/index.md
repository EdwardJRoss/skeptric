---
categories:
- programming
date: '2020-10-25T21:29:43+11:00'
image: /images/github_workflow_run.png
title: Manually Triggering Github Actions
---

I have been publishing this website using [Github Actions with Hugo](/github-actions) on push [and on a daily schedule](/github-action-cron).
I recently received an error notification via email from Github, and wanted to check whether it was an intermittent error.
Unfortunately I couldn't find anyway to rerun it manually; I would have to push again or wait.
Fortunately there's a way to *enable* manual reruns with `workflow_dispatch`.

There's a Github blog post on [enabling manual triggers with workflow_dispatch](https://github.blog/changelog/2020-07-06-github-actions-manual-triggers-with-workflow_dispatch/).
Essentially you just have to add `workflow_dispatch` to the `on` sections in the workflow yaml.
Once this is done then there's a UI element on the Workflow in the Actions section in Github that lets you "Run Workflow".

![Github Workflow Dispatch](/images/github_workflow_dispatch.png)

Here's how my workflow `on` section looks now, having added the last line to allow `workflow_dispatch`.

```
on:
  push:
    branches: [ master ]
  pull_request:
    branches: [ master ]
  schedule:
    - cron: '4 22 * * *'
  workflow_dispatch:
```