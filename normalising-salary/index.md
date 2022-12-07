---
categories:
- jobs
date: '2020-12-25T08:00:00+11:00'
image: /images/au_job_salary_init.png
title: Normalising Salary
---

Salary ranges come in many forms; how can we convert them to a common form?
A first approximation is to *annualise* them; it ignores the difference between full-time, part-time, and temporary work.

The other question is how to pick the range, for jobs with a bery large range.
I started with the minimum because the maximum is often an inspirational nubmer (especially in commission sales roles).

The way I approached this was:

* Look for anomalies (e.g. salaries where minimum is more than the maximum)
* For salaries with a period (e.g. hourly, daily, or annual) look at the range of common salaries
* Remove any data with wrong ranges due to issues in the data or in the parsing
* For salaries without a period, infer the period from the range (ignoring when it's ambiguous)
* Divide out the period (inferred or actual) to get the annualised salary.

For example by looking at the data I can see for Australian jobs annual salaries should be above \$10,000.
Daily salaries are above \$100 and hourly salaries below \$200; between \$100 and \$200 it's ambiguous depending on the kind of role.
But below \$100 it's unambiguously hourly.
This approach could be applied to different markets and currencies I'm less familiar with.

I used the [TDD approach to parsing salary](/tdd-salary), which allowed me to improve it and the tests caught some regressions I would have introduced.

After removing the out-of-band result and annualising I got a reasonable result:

![Distribution of annualised salary](/images/au_job_salary_after.png)

I undoubtedly removed some results that are valid, or that could be corrected, but this was an effective way of getting a lot of the valid data with a little work.
I've got a [Notebook showing the approach](/notebooks/Analysing\%20Salary\%20Extracted\%20From\%20CommonCrawl\%20Job\%20Data.html) ([raw](https://nbviewer.org/github/EdwardJRoss/skeptric/blob/master/static/notebooks/Analysing\%20Salary\%20Extracted\%20From\%20CommonCrawl\%20Job\%20Data.ipynb)).