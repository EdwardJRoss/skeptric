---
categories:
- programming
date: '2020-12-13T22:05:49+11:00'
image: /images/openhub-repositories-vcs.png
title: 'Git: One VCS to Rule Them All'
---

When I started as a professional developer there were a number of competing version control systems.
However Git seems to have [almost entirely](https://www.openhub.net/repositories/compare) won this battle.

One of the most popular centralised version control systems is Subversion (SVN), which [was largely an improvement](https://stackoverflow.com/questions/1261/what-are-the-advantages-of-using-svn-over-cvs) of Concurrent Versioning System (CVS).
But Distributed Version Control Systems, starting with Git became really popular.
With a centralised system you have to lock files on the central server when editing and unlock them when you're finished, to make sure no one else interferes with your work.
With a decentralised system you copy the whole repository locally and then merge together changes after.

Git made a lot of sense for Linux kernel development, for which it was originally developed.
Hundreds of developers were simultaneously trying to work on a codebase, many of whom were loosely coupled.
It made sense for them to work on pieces independently, submit their patches in email threads and have upstream maintainers merge them together and resolve any conflicts (where two people work on the same file).
Because of the size of the codebase and the social structure of Linux development it was very successful for them.

I'm really surprised distributed version control systems were so popular in corporations.
In a lot of companies small teams are working on a codebase in a highly coordinated way, regularly meeting to discuss development.
This means that decentralised development isn't going to be heavily utilised.
However there are some advantages; having local branching and commits allows experimentation that was a bit more expensive in CVS.

Git can now largely replace Subversion and has ways of working around some of the painful differences.
In Subversion you would only checkout the latest version of the code from the server.
With Git you get *all* the history which could be huge, but you can now specify [depth for a shallow checkout](https://git-scm.com/docs/git-clone#Documentation/git-clone.txt---depthltdepthgt).
For binary and media files you can't really merge them and are better off with a locking mechanism; but this is provided by [Git LFS](https://git-lfs.github.com/).

There were several competitors that were beaten by Git, most notably Mercurial (hg), but also Bazaar (bzr), and darcs.
Mercurial had a nicer user interface than Git, which was a bit more straight forward (but still complex, because distributed version control is complex) and more extensible.
But it wasn't *that* much better, and I think it lost in open source when Github (which only supports git) became much more popular than Bitbucket (which supported both).
Over time more projects moved onto Git (such as [emacs](https://www.masteringemacs.org/article/emacss-switch-to-git)), and there are only a few left on other DVCS.

I think it's largely good there's one mainstream version control system.
Any large software project has dozens of tools everyone has to be familiar with.
Git is a bit of a complex beast, but it's nearly everywhere so you can at least get payoff for your investment (although I hear in the gaming industry Perforce is more common).
There are good porcelains now that hide the complexity, including lots of GUI ones and the excellent [magit in Emacs](https://magit.vc/).
I've heard teams argue for weeks about which branching strategy to use (when really it didn't matter much in their case); having to *choose* a version control system is another argument they have to make.
Having a sensible default like Git is actually really helpful, one less decision to make.