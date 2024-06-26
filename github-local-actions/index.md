---
categories:
- programming
- blog
date: '2020-06-19T08:00:00+10:00'
image: /images/github_actions.png
title: Using Local Github Actions
---

I've been using Github Actions to [publish this website](/github-actions) for almost a month.
The experience has been great; whenever I push a commit it gets consistently published without me thinking about it within minutes.

However I have one concern; I'm passing my rsync credentials into an [external action](https://github.com/wei/rclone).
I've specified a tag in my yaml `uses: wei/rclone@v1`, but it would be easy for the author to [move this tag to another commit](https://github.com/wei/rclone) that sends my private credentials to their personal server.
Maybe I could specify the SHA-1 of the commit, but [SHA-1 is broken as a cryptographic hash](https://sha-mbles.github.io/) and someone may be able to forge a commit with the same SHA-1.

The safest thing to do is to move the action locally, which is very easy to do.

1. Copy the relevant [action code](https://github.com/wei/rclone) to `.github/actions/rclone` (or wherever you want in the repository)
2. Update the `uses` clause to the new destination prefixed by `./`: `uses: ./.github/actions/rclone`

This shows how easy it could be to make your own custom actions locally.
I like that it can be built on top of a Docker file, which makes it relatively straightforward to migrate to another CI/CD system.

Now I'm much more comfortable with how my secrets are being used.
I'm still trusting Github to store and pass my secrets correctly, I'm still downloading rclone from `https://rclone.org/install.sh` so I'm trusting that domain hasn't been acquired (and the [CA](https://en.wikipedia.org/wiki/Certificate_authority) certificates fetched by Alpine Linux are correctly validating the domain).
But it seems less risky than relying on some unknown Github repository with no reputation at stake (and I was already relying on all of those).

::: {.callout-note}
## Update

Since this was written rclone have released an [image on Dockerhub](https://hub.docker.com/r/rclone/rclone).
I'm now using this rather than downloading rclone, which transfers the trust from rclone.org to Dockerhub's credentials.
:::

I'm still comfortable using the external [Hugo Setup](https://github.com/marketplace/actions/hugo-setup) action because I don't pass any secrets.
