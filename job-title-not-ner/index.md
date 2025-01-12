---
categories:
- nlp
- data
- jobs
- python
date: '2020-04-06T08:03:17+10:00'
image: /images/job_titles_common.png
title: Not using NER for extracting Job Titles
---

I've been [trying to use Named Entity Recogniser (NER)](/ner-prodigy/) to extract [job titles from the titles of job ads](/job-title/) to better understand a collection of job ads.
While NER is great, it's not the right tool for this job, and I'm going to switch to a counting based approach.

NER models try to extract things like the names of people, places or products.
[SpaCy's NER model](https://spacy.io/universe/project/video-spacys-ner-model) which I used is optimised to these cases (looking at things like capitalisation of words).
A role title is actually a reasonable candidate for NER, but I was trying to make some fine grained linguistic distinctsion in my [annotation scheme](/job-title-annotation/).

However a job ad title is mostly the job title anyway, with some specific information like working conditions, seniority and location.
When I looked at the top 1600 titles that occurred accross many different advertisers I found that most of the titles were actually job titles with a few example with additionl words like "senior", "part time" and "immediate start".

It seems like by looking at the most common sequences of terms in job ad titles, blacklisting terms that aren't part of the title (like senior) is a good way to cover the most common job titles (which is the best I'd get out of NER anyway).
Then a first cut of labelling is just matching the strings.
Indeed published an [interesting pipeline](https://engineering.indeedblog.com/blog/2019/09/normalizing-resume-text-in-the-age-of-ninjas-rockstars-and-wizards/) for normalizing job titles.
This could be improved by performing some normalisation like expanding acronyms, stemming and grouping.

NER would be a great approach if the job title was hidden in a much bigger piece of prose, and in particular if I wanted to catch new and infrequent job titles as they came in.
But for summarisation frequency counting (with some normalisation) is a simpler approach that looks like it should work reasonably well.
The main concern is coverage; there will be some rare job titles that it misses (and NER may pick up), but from an aggregate viewpoint these aren't much value unless I can relate them to more common role titles.