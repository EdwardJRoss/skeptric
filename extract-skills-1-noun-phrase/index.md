---
categories:
- nlp
- jobs
date: 2020-01-07 09:30:08+11:00
image: /images/job_experience_noun_phrase.svg
title: 'Extracting Skills from Job Ads: Part 1 - Noun Phrases'
---

I'm trying to extract skills from job ads, using job ads in the [Adzuna Job Salary Predictions Kaggle Competition](https://www.kaggle.com/c/job-salary-prediction). Using rules to extract noun phrases ending in experience (e.g. *subsea cable engineering experience*) we can extract many skills, but there's a lot of false positives (e.g. *previous experience*)

You can see the [Jupyter notebook](/notebooks/Parsing Experience from Adzuna Job Ads.html) for the full analysis.

# Extracting Noun Phrases

It's common for ads to write something like "have *this kind of* experience":

* They will need someone who has at least 10-15 years of **subsea cable engineering experience**
* This position is ideally suited to high calibre engineering graduate with significant and appropriate **post graduate experience**.
* **Aerospace industry experience** would be advantageous covering aerostructures and/or aero engines.
* A sufficient and appropriate level of **building services** and **controls experience** gained within a client organisation, engineering consultancy or equipment supplier.

We can try to extract the type of experience using [spaCy's](https://spacy.io) `noun_chunk` iterator which uses [linguistic rules](https://github.com/explosion/spaCy/blob/v2.2.3/spacy/lang/en/syntax_iterators.py#L7) on the parse tree to extract noun phrases:


<!--Exported from Displacy-->
<blockquote>
<div class="entities" style="line-height: 2.5; direction: ltr">
<mark class="entity" style="background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    They
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">Noun Chunk</span>
</mark>
 will need
<mark class="entity" style="background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    someone
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">Noun Chunk</span>
</mark>
<mark class="entity" style="background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    who
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">Noun Chunk</span>
</mark>
 has
<mark class="entity" style="background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    at least 10-15 years
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">Noun Chunk</span>
</mark>
 of
<mark class="entity" style="background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    subsea cable engineering experience
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">Noun Chunk</span>
</mark>
</div>
</blockquote>


<!--Exported from Displacy-->
<blockquote>
<div class="output_subarea output_html rendered_html"><div class="entities" style="line-height: 2.5; direction: ltr">
<mark class="entity" style="background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    A sufficient and appropriate level
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">Noun Chunk</span>
</mark>
 of
<mark class="entity" style="background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    building services and controls experience
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">Noun Chunk</span>
</mark>
 gained within
<mark class="entity" style="background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    a client organisation
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">Noun Chunk</span>
</mark>
,
<mark class="entity" style="background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    engineering consultancy
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">Noun Chunk</span>
</mark>
 or
<mark class="entity" style="background: lightblue; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    equipment supplier
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">Noun Chunk</span>
</mark>
.</div></div>
</blockquote>

We can just look for all `noun_chunks` that end in experience, and grab every token leading up to experience

    def extract_noun_phrase_experience(doc):
        for np in doc.noun_chunks:
            if np[-1].lower_ == 'experience':
                if len(np) > 1:
                    yield 'EXPERIENCE', np[0].i, np[-1].i

## Analysing the results

Looking at the results from extracting the top fifty thousand job ads, the most common things it extracts aren't skills but *qualifiers* like "previous experience", "Proven experience", "some experience", and "demonstrable experience".

By filtering with a blacklist of the most common qualifying words, and stop words (the, this, an) we get some kinds of fields of expertise:

* sales
* management
* supervisory
* customer service
* development
* supervisory
* technical
* management
* telesales
* financial services
* design
* project management
* retail
* business sales
* SQL
* marketing
* people management
* SAP
* engineering

While this is a good start, building a longer list requires. building a much longer blacklist of qualifier terms (e.g. proven, demonstrable, demonstrated, relevant, significant, practical, essential, desirable, ...).
The fact that these qualifier terms are so common is because job ads commonly contain phrases like "previous experience in ..." or "some experience as ...".

In the [next post in the series](/extract-skills-2-adpositions/) we look at extracting from these types of phrases, and get much better results.