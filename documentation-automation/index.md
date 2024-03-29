---
categories:
- programming
date: '2020-12-09T19:01:30+11:00'
image: /images/automation.jpg
title: Automation through Documentation
---

You join a new team and your first task is to run the monthly batch process.
It transforms some data, trains a business critical model, and outputs some reporting.
Your coworker who is leaving the team talks you through the process and what you need to do.
The problem is that it's a bunch of scripts and ad hoc SQL that breaks all the time and has to be manually patched over.

The first temptation is to try to automate the whole process; but it's way too much.
And there are edge cases that only occur every 3rd or 12th month that you need to fix.
And between spending the two weeks each month running the process, and other work you just can't get the time to stop and automate the whole thing.

A better way forward is automating the process by documenting the steps.
As you run the process write down what you do.
When it breaks with this error, then refresh this table.
After running this process check these metrics to ensure this.

Each time you run the process follow and improve the documentation.
Fix any errors, and add instructions for any conditions to check.
Don't change the process too much, but if there's any low hanging fruit of obvious checks or fixes that could save a lot of stress/time put them in.

The next test is to see if someone else can follow the documentation without assistance.
Someone who doesn't have the context can follow the steps and complete the process.
Any gaps that they have to fill in that aren't immediately obvious should be added to the documentation.

Now you have a robust understanding of the process you can look for automation opportunities.
Remember that [automation should be Iron Man, not Ultron](https://queue.acm.org/detail.cfm?id=2841313); if you just automate the bits that are easy to automate you might leave the people with the really tricky edge cases that they don't have the skills to solve.
Try to make the automation transparent and not too clever.

It may take several months to get the whole thing working, but working in small increments will help you understand the process as you automate it rather than implementing a half-complete solution.
In general separating the doing from the things to be done, like in an issue tracker, allows you to prioritise better and not get too distracted fixing a bunch of things that aren't that important.
