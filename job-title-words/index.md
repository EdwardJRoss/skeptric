---
categories:
- jobs
- nlp
date: '2020-06-02T20:09:42+10:00'
image: /images/role_words.png
title: Job Title Words
---

I found [NER wasn't the right tool for extracting job titles](/job-title-not-ner), and a frequency based approach is going to work better.
The first step for this is to identify words that signify a job title, like "manager", "nurse" or "accountant".
I develop a whitelist of these terms and start moving towards a process for detecting role titles.

I have developed a method for [identifying duplicate job ads](/near-duplicate-review) and used it to remove duplicates.
This is because if one ad with the same title is posted hundreds of times (which happens in the dataset) it will have misleading results on the role title counts.
The process isn't perfect, but definitely reduces a lot of the noise without removing much signal.

Looking at the most common role titles by frequency they are terms like "Business Development Manager", "Project Manager", "Management Accountant", "Cleaner" and "Sales Executive".
That is they are have a *type* like "Manager", "Accountant", "Cleaner", or "Executive" and then a *specialisation* like "Business Development", "Project", "Management" or "Sales".
My approach is to try to build a whitelist of *types* to extract from the job text.

For a first step I get the top 800 most frequent role titles (where there"s 20 different job ads with that exact title modulo upper case) and count the number of ads by the last word from the `Title` field of my jobs dataframe `df`.

```python
roles = (
 df
 .Title
 .value_counts()
 .head(800)
 .to_frame()
 .reset_index()
 .assign(last_word=
     lambda df: (df['index']
                 .str.lower()
                 .str.split(' ')
                 .apply(lambda x: x[-1])))
 .groupby('last_word')
 .agg(n=('Title', 'sum'))
 .sort_values('n', ascending=False)
)
```

Then I manually went through this list and commented out the things that weren't general roles.
When I wasn't sure I looked back in the source data to see how that role was used; for example "partie" which is part of the special role title "chef de partie".

```python
df[df.Title.str.contaions('partie', case=False)].head()
```

I went through this list and removed anything that couldn't be a standalone role (e.g. "senior" was in but "partie" was out), anything that was an industry rather than a role (e.g. "finance") and plurals (e.g. "nurses") and roles containing a slash (like "receptionist/administrator").
This left me with a list of 104 role types (like manager, engineer, executive, assistant, accountant, administrator, ...).
"Sales" is an interesting exception because sometimes it's used as a *type* (e.g. medical sales), sometimes as a *specialisation* (e.g. sales executive) and sometimes as an *industry* (e.g. head of sales).
There are also many that are only *types* in certain context, like "assistant" in "sales *assistant*", but not in "*assistant* manager".

In building this list I also I also built a mapping of acronyms (like PA is Personal Assistant, DBA is Database Administrator).
Once these are expanded you get to standard types like "assistant" and "administrator".
I also noticed adviser is a spelling variant of advisor.

Looking into the role titles that end in an industry (like "finance", or "marketing") they are mainly of the form "director of marketing", "head of finance", or "teacher of English".
We can generally consider these as equivalent to "marketing director", "finance head" and "English teacher".
In this case they also end in standard role types like "director" and "teacher".
Head is a rather uncommon ending, but it *does* occur in the source data like "Group Head" and "Department Head".

This first look at the data gives an idea of an approach.
We can expand acronyms, normalise spelling variants, and then try to match on a type.
There are still some challenges like how to deal with slashes (e.g. fitter/turner), multiple types together, and things that are a type depending on context like assistant or head.
There's still a challenge with how to normalise something like "director of marketing and finance" or an overly specific role title like "financial planning and analysis manager".
But these problems are manageable, and more the edge cases than the common cases.

For more details see the [Jupyter notebook](https://github.com/EdwardJRoss/job-advert-analysis/blob/master/notebooks/Extracting%20Role%20Title%20Words.ipynb).