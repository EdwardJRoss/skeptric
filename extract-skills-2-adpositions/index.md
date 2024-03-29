---
categories:
- nlp
- jobs
date: 2020-01-13 16:40:20+11:00
image: /images/job_experience_adposition.svg
title: 'Extracting Skills from Job Ads: Part 2 - Adpositions'
---

# Extracting Experience in a Field

I'm trying to extract skills from job ads, using job ads in the [Adzuna Job Salary Predictions Kaggle Competition](https://www.kaggle.com/c/job-salary-prediction).

In the [previous post](/extract-skills-1-noun-phrase/) I extracted skills written in phrases like "subsea cable engineering experience".
This worked well, but extracted a lot of qualifiers that aren't skills (like "previous experience in", or "any experience in").
Here we will write rules to extract experience from phrases like "experience in subsea cable engineering", with much better results.

You can see the [Jupyter notebook](/notebooks/Parsing Experience from Adzuna Job Ads.html) for the full analysis.

# Extracting experience in something

By looking at the parse trees of candidate phrases using [displaCy](https://explosion.ai/demos/displacy?text=Experience%20in%20Modelling%20and%20Simulation%20Techniques&model=en_core_web_sm&cpu=1&cph=0) I adopted the following strategy:

![Example Dependency Parse](/images/job_experience_adposition.svg)

* Start at the word experience, and look for a preposition (such as in or of) dependent on it (red in the above diagram)
* Look for the object of the preposition (orange)
* Return the phrase ending at that object (green)

or in code:

    def extract_adp_experience(doc):
        for tok in doc:
            if tok.lower_ == 'experience':
                for child in tok.rights:
                    if child.dep_ == 'prep':
                        for obj in child.children:
                            if obj.dep_ == 'pobj':
                                yield 'EXPERIENCE', obj.left_edge.i, obj.i+1


A simpler way to do this is:

* Start at the word experience followed by a preposition (such as in, of, or with)
* Get the noun phrase following it

Using spaCy's noun chunks we have to implement this backwards:

    def extract_adp_experience_2(doc):
        for np in doc.noun_chunks:
            start_tok = np[0].i
            if start_tok >= 2 and doc[start_tok - 2].lower_ == 'experience' and doc[start_tok - 1].pos_ == 'ADP':
                yield 'EXPERIENCE', start_tok, start_tok + len(np)

Both algorithms give similar results, so there's some flexibility in how you write the extraction rules.

We could try to further extend the rules with examples where there's an extra level of indirection, such as:

>    Previous experience working as a Chef de Partie in a one AA Rosette hotel is needed for the position.

>    Experience of techniques such as Discrete Event Simulation and/or SD modelling Mathematical/scientific background

>    The post holder must hold as a minimum Level 1 in Trampolining (British Gymnastics) and have experience in working with children, be fun, outgoing and have excellent customer service skills and be able to instruct in line with the British Gymnastics syllabus.

but the rules become increasingly complex and aren't likely to add much to the results.

# Analysing the Results

A company sometimes posts a job ad many times with very similar text, so it makes more sense to rank results by the number of distinct companies that posted the term rather than the number of times it occurs alone.

While the top terms contains some generic phrases (like "a similar role" or "the following"), it also contains a lot of genuine skills like "design", "C", "selling", and "project management".
The broader skills like "design" and "selling" often have a qualifier that we are not extracting (e.g. "selling into the industrial sector" is different to "selling into the veterinary/animal industry"), but it's a pretty good start.

Looking at the first 50,000 ads here are the top 30 extracted skills:


| Term | Number of Companies | Number of Occurrences |
|------|-----------|---|
| a similar role | 213 | 461 |
| the following | 130 | 261 |
| sales | 77 | 106 |
| one | 55 | 85 |
| the design | 53 | 83 |
| the use | 49 | 72 |
| design | 47 | 76 |
| C | 46 | 87 |
| selling | 43 | 60 |
| this role | 42 | 87 |
| all aspects | 40 | 66 |
| this | 39 | 58 |
| experience | 38 | 55 |
| the following areas | 37 | 65 |
| planning | 37 | 46 |
| teaching | 34 | 63 |
| any | 34 | 54 |
| development | 34 | 56 |
| project management | 34 | 49 |
| this field | 33 | 58 |
| the industry | 33 | 51 |
| a manufacturing environment | 31 | 46 |
| SQL Server | 30 | 57 |
| software development | 29 | 46 |
| a | 28 | 50 |
| some | 28 | 44 |
| a similar environment | 28 | 42 |
| SQL | 28 | 35 |
| this area | 27 | 47 |

# Relating different skills

It would be really interesting to see which skills occur together, but ads aren't likely to contain the phrase "experience in/with" many times, and so we're not likely to extract many skills from a single ad.
However ads frequently list experience in long lists, for example "Experience in design, development or quality engineering".

In the [next part](/extract-skills-3-conjugations) we will extract these phrases and investigate what skills frequently occur together.