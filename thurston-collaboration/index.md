---
categories:
- maths
date: '2020-05-06T07:59:26+10:00'
image: /images/thurston.jpg
title: Lessons from a mathematician on building a community
---

Mathematicians and software developers have a lot in common.
They both build structures of ideas, typically working in small groups or alone, but leveraging structures built by others.
For software developers the ideas are concrete code implementations, and the building blocks are subroutines, and are published as "libraries" or "packages".
For mathematicians the ideas are abstract, built on definitions and theorems and published in papers, conferences and informal conversations.
To grow a substantial body of work in both mathematics or software requires a community to contribute to it.

The best way to get people to contribute is to make it easy for them to contribute.
As a contributor it can be tempting to complete a product to a nice finish, but this makes it hard for other people to get involved.
Instead if you leave obvious gaps for other people to fill it helps them get involved, understand the work and feel ownership.
Some successful open source projects do this by tagging issues as "good first issue", and leaving them for other people to tackle.

The mathematician [William Thurston](https://en.wikipedia.org/wiki/William_Thurston) wrote about his experience of "completing" in his excellent article [On proof and progress in mathematics](https://arxiv.org/abs/math/9404236):

> At that time, foliations had become a big center of attention among geometric topologists, dynamical systems people, and differential geometers.
> I fairly rapidly proved some dramatic theorems...
> I wrote respectable papers and published at least the most important theorems.
>
> An interesting phenomenon occurred.
> Within a couple of years, a dramatic evacuation of the field started to take place.
> I heard from a number of mathematicians that they were giving or receiving advice not to go into foliations - they were saying that Thurston was cleaning it out.
> People told me (not as a complaint, but as a compliment) that I was killing the field.
> Graduate students stopped studying foliations, and fairly soon, I turned to other interests as well.
>
> I do not think that the evacuation occurred because the territory was intellectually exhausted - there were (and still are) many interesting questions that remained that are probably approachable.
> Since those years, there have been interesting developments carried out by the few people who stayed in the field or who entered the field, and there have also been important developments in neighbouring areas that I think would have been much accelerated had mathematicians continued to pursue foliation theory vigorously.
>
> Today, I think there are few mathematicians who understand anything approaching the state of the art of foliations as it lived at that time, although there are some parts of the theory of foliations, including developments since that time, that are still thriving.

He goes on to speculate that the way he wrote his papers, in a typical dense mathematical style, making references to many different fields created a high entry barrier.
This made it hard for new graduates to enter the field to keep it grow.
The other major problem is by proving the big theorems he didn't leave room for other people to enter the field and start building an understanding.

> More than the knowledge, people want personal understanding.
> And in our credit-driven system, they also want and need theorem-credits.

This can equally apply to developing software.
Most of the time I find reading new software painful; you have to build a mental model of how everything fits together and the individual pieces work.
It's even harder if it's in a language or uses libraries that I'm not familiar with.
So if some code does most things I want without any issues I'll happily use it without looking at the source code to see how it works, and I can always build on top of it if I need to.
This is similar to the mathematicians who could happily use and accept Thurston's proofs without understanding them or contributing to the field.

This is why it's very impressive when there's a thriving open source project that requires specific technical expertise or diverse programming languages like [Apache Arrow](https://arrow.apache.org/) - I have no idea how they do it!

Thurston also talks about how he learned from his experience on foliations, and when he worked on 3-manifolds he took a different approach.
He spent a number of years understanding the field before he proved a major result and made a [conjecture](https://en.wikipedia.org/wiki/Geometrization_conjecture).
He taught the underlying ideas to his graduate class, and started providing the notes to a mailing list that grew to 1200 people (which is very large in mathematics!).
He presented seminars and workshops, helping the community understand the "infrastructure" in the proof that spanned different areas of mathematics.
This then helped grow the field by giving the tools and space for other people to make in-roads into the field and publish their own papers (and eventually led to solving a [100 year old problem](https://en.wikipedia.org/wiki/Poincar%C3%A9_conjecture)).

> There has been and there continues to be a great deal of thriving mathematical activity.
> By concentrating on building the infrastructure and explaining and publishing definitions and ways of thinking but being slow in stating or in publishing proofs of all the “theorems” I knew how to prove, I left room for many other people to pick up credit.

I've had a smaller experience in using these ideas in building software when I was working on custom reporting.
We had some PHP scripts that could build the simplest reports, but for most reports we would need to hand-roll them in Python because of slightly different requirements (the PHP scripts weren't modular enough to build libraries on; they were very brittle).
This was a lot of copy-paste-edit, but meant each reporting script required separate QA, and once in a while they would fail on a corner case that was solved in a different variation of the script.

A very talented colleague rewrote one of the reporting libraries in Python.
It covered the use case very well, but was quite complex, building up a large SQL query through several code paths depending on the use case.
This meant it was difficult to extend; it took me a few iterations because it was difficult to keep the queries and their interactions in my head.

I tried to build a single library that would solve all the use cases (which *would* have looked like a half-baked [dbplyr](https://dbplyr.tidyverse.org/) if it had worked), but after a couple weeks of work got very stuck - it was beyond my skill level.

A month or so later when I had a complex report I built a library to tackle the simplest (and most frequently used) report.
It used some of the ideas from the library that had already been rewritten, but was much simpler and tended to do more in Python (which was more flexible) than SQL (which was faster to execute).
I made sure it could be imported as a module to use in bespoke reporting code, as well as having a CLI interface for simple reports, and would cover the use case completely.
Being able to import the script in Python made it easy to build complex reports around and it quickly got adoption in the team.

While it was clear that building more of these reporting scripts would be beneficial, it was always hard to justify the upfront cost of writing it rather than working on actual reports.
Because it had to be robust it often took much longer than individual reports where you could skirt around the edge cases and only deal with one way of reporting.
There were also ongoing maintenance costs which made it harder to write more code, when I only had the existing reports.

I really needed more help from my colleagues, but I was the only person who was maintaining and extending the codebase.
So when I built the next library I took a leaf out of Thurston's book; I thought hard about what needed to be done, but I didn't do all of it.
In particular the library would not work on some well documented cases, but that were relatively easy to add.
This meant it was useful enough to use in some reporting scripts, but for the more complex ones you would have to write some custom queries.

It took a few months (and some gentle encouragement), but eventually one of my colleagues had this kind of complex report and extended the library to cover this use case.
It was a small enough piece of work to do in a couple of days, and I could help guide him on what needed to be done and review the fix.

This kind of cycle continued and the person who extended this library went on to write a couple of others.
Eventually most of the team contributed to this growing codebase, and we covered almost all the common cases.
Some other great ideas were introduced; someone introduced the team to [Pandas](https://pandas.pydata.org/) and putting it on top of the libraries meant we could write more complex reports more quickly.

Reducing the reporting workload also gave us more "blue sky" time to innovate, including building new profitable products and a self-serve reporting interface, which saved even *more* time.

Looking back, it was really hard to get anyone else to work on the library I had started on.
However by having some obvious gaps that were simple to fill led someone to make the first step and get familiar with the codebase and have some ownership.
This let us build out more usecases and help get more of the team involved, until most of the team was able to contribute and maintain it.
We were fortunate in that once we had a clear frame for the API it was clear what needed to be done, and it could be immediately used in our client work.

If you're trying to build a substantial piece of software it's a lot easier with a community.
Take on Thurston's lessons, start building something useful, spend a lot of effort educating about it, leave obvious gaps for contributions and do everything you can to support those contributions.
It takes a long time to build understanding of software, but if you can get people to invest in contributing they will be incentivised to slowly build that understanding and contribute more.

I'd be really interested in understanding more how people who are really good at building communities like Hadley Wickham, Wes McKinney and Linus Torvalds (to name a few large open source project maintainers) do it.