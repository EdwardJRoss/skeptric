---
categories:
- jobs
- nlp
date: '2020-06-09T08:00:04+10:00'
image: /images/common_role_titles.png
title: Discovering Job Titles
---

A job ad title can [contain a lot of things](/job-title) like location, skills or benefits.
I want a list of just the job titles, without the rest of those things.
This is a key piece of information extraction that can be used to better understand jobs, and built on by understanding how different job titles relate, for example with [salary](/job-ad-title-salary).

To do this we first [normalise the words in the ad title](/normalise-job-title-words), doing things like removing plurals and expanding acronyms.
Then we search for common [role words](/job-title-words) like "manager", "nurse" or "accountant"; I've got a list of 120 of them.
Then we look backwards to expand the role; e.g. from "manager" we could get to "account manager" or "product manager" or "software engineering manager".
Then we sort them by frequency, blacklist any ambiguous ones, and have a list of common role titles.

This works reasonably well, with a couple iterations on the ambiguous word list I get a long list of common titles.
There are a couple of exceptional role titles that don't fall in this order like "chef de partie" which I need to add separately (it starts with the most general part and then gets specific, compared to most role titles that start specific and then become generic).

The job website Indeed have a [more aggresive pipeline for finding role titles](https://engineering.indeedblog.com/blog/2019/09/normalizing-resume-text-in-the-age-of-ninjas-rockstars-and-wizards/).
They just look for all groups of frequent terms in the titles after normalising the text.
I would worry I then need to somehow remove phrases containing common words that aren't job titles; like locations (e.g. London) or skills and benefits.
They then further group these together using [edit distance](/levenshtein), so "manager" and "manger" would go together, as would "C# developer" and "C developer" (which should be different role titles).
I definitely miss unusually phrased job titles with my method (e.g. "Engineering role"), but I get less false positives.

The implementation of my approach is very straightforward, I use a regular expression to find a role word (or an exception) with up to four preceding words.
Four is roughly where I see diminishing returns on looking for longer role lengths, five would be another to try.

```python
role_word_re = r'\b(?:' + '|'.join(exceptions + role_words) + r')\b'
preceding_word = r'(?:\b[\w\d]+\s+)'
role_term_re = re.compile(preceding_word + '{0,4}' + role_word_re)
def find_maximal_roles(title):
    normal_title = normalise_text(title)
    return role_term_re.findall(normal_title)
```

This regular expression approach does have some warts, because regular expressions must be non-overlapping.
For example "PA to CEO of top finance boutique" after expansion becomes "personal assistant to chief executive officer of top finance botique", which gives roles "personal assistant to chief executive" and "officer".
Or "Assistant Head of Planning and Building Control" gets normalised to "planning and building control assistant head" which then gives roles "planning and building control assistant" and "head".
But it's simple to implement so is a useful starting point.

Looking through the roles there were a few ambiguous ones like "head", "senior", "officer" and "specialist" that were fine if they were part of a longer title, (e.g "finance head" or "semi senior"; which is apparently an accounting thing) but on their own were normally the result of a bad normalisation.
After filtering these out I got a list of 1000 viable role titles that occurred in at least 20 ads.
Some of these include seniority (like "head chef") or work type (like "part time cleaner"), but are a very useful starting point for further analysis, such as of [salary](/job-ad-title-salary).

For further details see the [Jupyter notebook](https://github.com/EdwardJRoss/job-advert-analysis/blob/master/notebooks/Extracting%20Role%20Titles%20and%20Analysing%20with%20Salary.ipynb).