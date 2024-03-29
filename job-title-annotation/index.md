---
categories:
- nlp
- data
- jobs
date: '2020-04-01T22:06:56+11:00'
image: /images/annotated_title.png
title: Annotating Job Titles
---

When doing Named Entity Recognition it's important to think about how to set up the problem.
There's a balance between what you're trying to achieve and what the algorithm can do easily.
Coming up with an annotation scheme is hard, because as soon as you start annotating you notice lots of edge cases.
This post will go through an example with extracting job titles from job ads.

In our [previous post](/job-title) we looked at what was in a job ad title and a way of extracting some common job titles from the ads.
There's no obvious programmatic rules that could be constructed for extracting the job title from the title of the advertisement, so one approach is to train a Named Entity Recogniser.
To do this we need to annotate many examples of a job title, which means we need to be crystal clear on what *is* a job title.


We definitely don't want the location of the job, the name of the company or the working conditions (like "Part Time", "Excellent Tips"); these are separate pieces of information.
However the seniority and industry *may* sometimes be relevant, and hard to separate.

For example a "Pharmaceutical Sales Representative" may be a different type of job to a general "Sales Representative" because it requires different qualifications and skills.
You could make arguments that the job title for "Subsea Cabling Engineer" could be "Subsea Cabling Engineer", "Cabling Engineer" or just "Engineer".
These distinctions are very subtle and require a lot of world knowledge - not a good fit for a NER algorithm.
The best thing to do is try to capture the longest role title (like "Subsea Cabling Engineer") and then further break it down with post-processing.

Of course when you start looking at the data there are many subtle examples.
How about when the industry is tacked onto the end like: "Quality Manager  Defence"?
Is a "Web Programmer - C#, ASP.NET" different to a "C#/ASP.NET Web Programmer"?
It's really hard to be consistent in annotation (and if we can't agree on the correct answer, how could a model?)

For terms like "Head of Marketing" and "Marketing Assistant" it's a little tricky to cleave off the seniority.
So similarly it makes sense to capture all the seniority with NER.
If we need to remove the title later we can do this with post-processing rules.
 
For joint roles like "Receptionist/Administrator" or "CNC Setter / Operator" again it makes sense to capture the whole phrase.
For example "CNC Setter / Operator" is really short for "CNC Setter / CNC Operator" but it's going to be tricky to do those expansions.
We can then choose to fix them up with post-processing.

So the goal for our NER is to capture the longest contiguous phrase containing roles, industries or seniorities.
This is still doing useful things like removing working conditions like "Part Time", company information, and location.
Then we can try to further break up the more structured roles with post-processing rules.

It sounds clear right?
But as soon as I start looking at examples I start doubting my choices!
"Planning Engineer - Civil Engineering  Derby", should I *really* include "Civil Engineering" in the role title?
How about "PA to a Leading Partner in a US Law Firm"?
By my own rules I should include "Law", maybe the "Leading Partner" changes the nature of the role?
Maybe here "Leading Partner" is a separate role title I should annotate!
I guess "Head of Care  Nursing (RN)" is a single title?

You really want a clear annotation scheme for your machine learning algorithm to learn.
But it's really hard to know what you want until you've spent a long time looking at the data.
At the end of the day I guess you have to spend a little time upfront defining an annotation scheme and then test and iterate to see whether it works in practice.

I'll see how this goes in future posts in this series!