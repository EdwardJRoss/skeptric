---
categories:
- linux
date: '2020-09-10T08:00:00+10:00'
image: /images/git_stash.png
title: Git Stash Changesets
---

Pretty frequently I start writing some code, when I realise there's another change I need to make before I can continue.
I like to make lots of small atomic changes to a code base because it lets me test more quickly and catch errors earlier.
I used to do this by saving my changes in a temporary file, but this was clunky.
A better way is with git stash.

But git stash reverts *all* files; and very often I want to keep some, especially configuration parameters.
However there's a way to stash just the files added for commit; use `git stash --index`.
The name comes from the `index` being the list of changes that are to be added for the next commit.
I use this in Emacs via [magit](https://magit.vc/manual/magit/Stashing.html) which lets me easily stash chunks.

The only downside of this is it won't let me pop the stash until the file is at the latest commit.
This means if I want to combine the changes in a single commit I have to commit, then pop the stash and amend the commit.
Not awful, just a little finicky.
But it's a pretty safe way to work.

Another alternative to stashes is branches.
It's very easy to make a temporary branch to store some work and move around.
But stashes require a bit less context switching, and are convenient to use.