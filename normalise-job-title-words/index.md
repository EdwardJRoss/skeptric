---
categories:
- jobs
- nlp
date: '2020-06-08T15:57:57+10:00'
image: /images/normalising_job_words.png
title: Normalise Job Title Words
---

I'm trying to find job titles in job ads, but the same title can be written lots of different ways.
An "RN" is the same as a "Registered Nurse", and broadly the same role as "Registered nurses".
As a preprocessing step to [job title discovery](/discovering-job-titles) I need to *normalise* the text.

The process I use is simple:

1. [rewrite](rewrite-of) terms containing of, e.g. "Director of Sales" to "Sales Director" 
2. Expand punctuation with whitespace; e.g. "Receptionist/Administrator" to "Receptionist / Administrator"
3. [Singularize](/making-words-singular) each word; e.g. "Cleaners" to "Cleaner"
4. Expand known acronyms; e.g. "DBA" becomes "Database Administrator"
5. Lowercase the text; e.g. "iOS Developer" becomes "ios developer"
6. Replace common variants and misspellings; e.g. "Adviser" becomes "Advisor"
7. Replace multiple whitespace with a single space

In code this looks like:

```python
def normalise_text(text, acronyms=None, variants=None):
    text = rewrite_of(text)
    text = expand_punctuation(text)
    text = singularize(text)
    text = expand_acronym(text, acronyms)
    text = text.lower()
    text = expand_acronym(text, variants)
    text = compress_whitespace(text)
    return text
```

This works reasonably well, but the order really matters and it's a little bit fragile.
We need to `expand_punctuation` before we `singularize`, because `singularize` tokenizes on spaces and so "Receptionists/Administrators" would be singularized to "Receptionists/Administrator", but "Receptionists / Administrators" would correctly transform to "Receptionist / Administrator".
We need to `singularize` before we `expand_acronym`, so that for example `RGNs` can be transformed to `RGN` before expanding to `Registered General Nurse`.
We need to lowercase before expanding variants, because they apply for any casing.

There are still some cases where this goes wrong, like ENGINEERS will not be singularised correctly unless we insert a second round of singularisation after lower casing.
But it does a pretty reasonable job.

We could be a lot more aggressive with out normalisation, in particular we could use stemming instead of making words singular, drop stop words and remove all punctuation.
However this would potentially lose some useful linguistic information, and I would rather gradually remove these as needed (by examining output data) rather than doing it all up front.

The rest of this article goes over each piece, except for [`rewrite_of`](/rewrite-of) and [`singularize`](/making-words-singular) which are covered in their own articles

# Expanding Punctuation

This is a simple process of putting extra space around each punctuation mark.
This helps downstream processes that rely on processing separate tokens work.
One caveat is we need to be careful with punctuation that can be part of an acronym (like A&E).
Perhaps it would be safer to do this just at the boundaries of words; but in practice this works well enough on the Adzuna dataset.

```python
EXP_PUNC_RE = re.compile('([/()\'":,])+')
def expand_punctuation(text):
    return EXP_PUNC_RE.sub(r' \1 ', text)
```

# Expanding Acronyms

There are lots of common acronyms that need to be expanded to be matched.
This is a simple process of substituting at word boundaries.

```python
def expand_acronym(title, acronyms):
    for source, target in acronyms.items():
        title = re.sub(fr'\b{source}\b', target, title)
    return title
```

Here's the list of acronyms I used:

```python
acronyms = {
    'PA': 'Personal Assistant',
    'DBA': 'Database Administrator',
    'RGN': 'Registered General Nurse',
    'RMN': 'Registered Mental Health Nurse',
    'NQT': 'Newly Qualified Teacher',
    'CEO': 'Chief Executive Officer',
    'MD': 'Managing Director', # Medical doctor doesn't occur often here
    'EA': 'Executive Assistant',
    'GP': 'General Practitioner',
    'ODP': 'Operating Department Practitioner',
    'A&E': 'Accident and Emergency',
}
```

# Normalising variants

Spelling variants is a very similar problem to acronyms, picking a common target way of writing something.
I reuse the `expand_acronyms` function on a list of variants; I do this after lowercasing to get common forms.

```python
variants = {
    'adviser': 'advisor',
    'draughtsman': 'draughtsperson',
    'registered mental nurse': 'registered mental health nurse',
    'comm': 'communication'
}
```

# Compressing whitespace

This is a simple procedure to remove any extra whitespace.

```python
WHITESPACE_RE = re.compile(r'\s+')
def compress_whitespace(text):
    return WHITESPACE_RE.sub(r' ', text)
```