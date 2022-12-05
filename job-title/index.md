---
categories:
- nlp
- data
- jobs
date: '2020-03-31T22:03:00+11:00'
image: /images/job_titles.png
title: What's in a Job Ad Title?
---

The job title should succinctly summarise what the role is about, so it should tell you a lot about the role.
However in practice job titles can range from very broad to very narrow, be obscure or acronym-laden and even hard to nail down.
They're even hard to extract from a job ad's title - which is what I'll focus on in this series.

In a [previous series](/extract-skills-1-noun-phrase/) of posts I developed a method that could extract skills written a very particular way.
Creating a broader extraction methodology is challenging (what is a skill?), so I thought I'd start with something easier: job titles.

So I'm trying to extract job titles from the titles of job ads in the [Adzuna Job Salary Predictions Kaggle Competition](https://www.kaggle.com/c/job-salary-prediction).
See the [Jupyter Notebook](/notebooks/Extracting Role Titles from Adzuna Ads.html) for the details of this analysis.


# Components of a job ad title

Most job ad titles have a job title like "Sous Chef", "Registered Nurse" or "Project Manager" (although some have none like "Immediate Start Due to Expansion").
But they also typically have other information:

* Seniority: "Senior", "Principal", "Lead", or "Trainee"
* Location: "Glasgow" or "East Midlands"
* Industry: "Pharmaceutical", "Construction",
* Working Conditions: "Part Time", "Award Winning Restaurant", "Excellent Tips", "Self Employed", "does it get any better than this?"
* Company names: "Nevill Crest and Gun", "The Refectory"

To capture the job ad title we want to try to remove these, but it's difficult because there's no easy way to identify all of these categories.
They're also not quite mutually exclusive.

# What is a job title?

There's lots of examples where the job title is ambiguous under the scheme above.

## Seniority

It's not always clear that separating seniority is the right thing to do.
We can easily split a "Senior Staff Nurse" into a "Senior" "Staff Nurse".
A "Marketing Assistant" may be able to be broken into an "Assistant" "Marketing Officer".
But is a "Marketing Manager" a "Manager" level "Marketing Officer"?
What about the "Head of Marketing"?

With seniority we need to make a call of how to split it off.

There are also some odd things that *may* be considered seniority.
In "Part Qualified Accountant" is "Part Qualified" a seniority?
How about "experienced" like in "Experienced Recruitment Consultan?
Or a "lunchtime supervisor" is presumably more junior than a "supervisor", but I don't know if we can consider "lunchtime" a seniority.

Seniority is often implicit in the role; a "Key Account Manager" is more senior than a "Regional Account Manager".
A "Human Resources Business Partner" is more senior than a "Human Resources Officer" which is more senior than a "Human Resources Administrator".
Trying to extract the seniority in general isn't straight forward.

## Multiple roles

Sometimes it's hard to work out how many roles there are in an ad title.

Is a "Receptionist/Administrator" one role or two?
Perhaps a "Receptionist/Administrator" does different kinds of work to a sole "Receptionist" and a sole "Administrator" so we could consider it a single role.
But it could be valid to consider it a mix of two different roles.

How about a "CNC Setter / Operator"?
I believe this is just one role covering two pieces of work using a CNC machine.

What about "Registered General Nuse/Registered Mental Health Nurse"?
It seems likely they'd prefer a [Registered Mental Health Nurse](https://www.nurses.co.uk/nursing/blog/how-to-become-a-mental-health-nurse-rmn/) since it's more specific, but will settle for a Reginistered General Nurse.
So I guess this is two titles because they would hire one of either (but more the latter!).


## Relating Roles

Even once we've found a role title it's hard to fit it into the constellation of roles.

Common acronyms can be expanded to see that a "RGN" is the same as a "Registered General Nurse", and similar things can be normalised to it.

But is a "groundworker" similar to a "gardener"?
A "reward Analyst" is likely more similar to a "marketing analyst" than an "infrastructure analyst"
Understanding the roles requires a lot of additional knowledge about the roles, and will require actually looking into the ad text.
But just *having* the role titles would be a useful start.

# How to extract role titles?

While we're not *totally* clear on what a role title is, we need to start making some progress to extract them.

Some advertisers do write very generic ad titles like "Project Manager" which are close to job titles.
The more advertisers that have the exact same ad title the more likely it is that it's a generic job title not containing specifics about the role like location and conditions.

I extracted all job ad titles that occurred in at least 10 advertisers.
This cutoff sound high, but because Adzuna aggregates ads from different job boards and they sometimes have different names for the same advertiser at a lower cutoff there were many examples of the same job.
We could lowercase the titles to get more examples, but I was interested in how casing was actually used.

I output the 1600 resulting titles into a CSV for further analysis, to see in the next part of the series.