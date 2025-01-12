---
categories:
- programming
date: '2020-11-19T21:27:14+11:00'
image: /images/sky-spaghetti.jpg
title: Operating a Tower of Hacks
---

Remember after you run the update process to run the fix script on the production database.
But run it twice because it only fixes some of the rows the first time.
Oh, and don't use the old importer tool in the import directory, use the one in the scripts directory now.
You already used the old one?
It's ok, just manually alter the production database with this gnarly query.
Ah right, I see the filler table it uses is corrupted, let's just copy it from a backup.

This may sound ridiculous but sometimes I've had situations that feel like this.
Things get built under operational pressure and you just have to get it working.
But when you wake up in a cold sweat at 5am because your pager didn't go off overnight, and so you assume the alerting system must be broken then its gone way too far.

It can be hard to get investment to stabilise these hacky systems, but leaving them as they are is extremely stressful.
I've found it useful to call out the operational risks, go through scenarios of the potential impact of near misses, and push back on other priorities because I'm busy trying to keep this system running.

One useful thing step can be to take the time to record each step of the process, including each hacky fix.
Once you've done this enough times to hit the edge cases you can hand this off to anyone else.
It becomes clearer where the really bad parts are, and helps prioritise the fixes.