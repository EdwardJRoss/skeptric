---
categories:
- general
date: '2020-11-14T21:20:40+11:00'
image: /images/mvp_path.png
title: Success in Small Steps
---

A lot of times I've failed by biting off more than I can chew.
I get in over my head and lose motivation.
A lot of times I've succeeded it's by starting small and slowly building up a roll of successes.

When I was in highschool I tried to build a simulation of the solar system for a project.
I wasn't satisfied with building ellipses, I wanted to take into account all the N-body interactions.
It started off alright, but quickly I was out of my depth trying to model all the bodies and got the code in a broken state I couldn't recover.
I ended up with pages of code that didn't work at all.
If I had started smaller with building a single elliptic orbit, and then another, before even considering multibody interactions I would at least have had a good state to revert to, and would have been more likely to succeed with the complex problem.

When I returned from a long overseas trip I was looking for work.
I wanted to build something to help showcase my skills for prospective employers.
I had the idea of building a Javascript illustration of a chaotic system; where a small change in starting points results in a very large difference in outputs.
I spent a lot of time learning Javascript, agonising over whether I use SVG or the canvas for performance, writing vectorised functions so that I could extend it into 3 dimensions, and so on.
I never actually built a functioning application because I got so bogged down on building it the *right* way; thankfully my CV and cover letters got me sufficient interviews.
I would give myself the advice to build something simple; even programming a moving ball would be a success I could showcase.
Then I could slowly build on that, adding a wall, adding bounce physics, adding angles. 
I tried to build a physics engine while learning web technology at the same time, it was too hard for me and I had nothing to show for it.

![The kind of chaotic](/images/chaotic_bunimovich_stadium.png)

This blog has examples of taking small steps.
The [jobs](/tags/jobs) posts show lots of small steps in trying to build a pipeline for extracting information from job adverts on the internet.
For example I had a post on [extracting job title words from ad titles](/job-title-words/) (e.g. "Manager" or "Executive" or "Engineer").
I wanted to improve this so I wrote follow ups on [making plural words singular](/making-words-singular/), and another on [rewriting "Head of Marketing" to "Marketing Head"](/rewrite-of/), and putting them together in a [normalisation strategy](/normalise-job-title-words/) which created a better way to [discover job titles in a dataset](/discovering-job-titles/).
The initial rough approach gave me confidence that the system could work, and then I could build upon that and improve it.

This is why I recommend starting with [simple models](/simple-models) and working towards a complex solution.
Whenever I've tried to start with a complex model I've spent too much time trying to get out and engineer features in the data without actually understanding it.
Now I start with building an evaluation criteria, starting with a simple model and slowly building on it.
Adding features to get incrementally better, and learning more about the data at the same time.