---
categories:
- nlp
- data
- jobs
date: '2020-05-31T06:24:48+10:00'
image: /images/ads_process_project_engineer.png
title: Summary of Finding Near Duplicates in Job Ads
---

I've been trying to find near duplicate job ads in the [Adzuna Job Salary Predictions Kaggle Competition](https://www.kaggle.com/c/job-salary-prediction).
Job ads can be duplicated because a hirer posts the same ad multiple times to a job board, or to multiple job boards.
Finding [exact duplicates](/exact-duplicates) is easy by sorting the job ads or a hash of them.
But the job board may mangle the text in some way, or add its own footer, or the hirer might change a word or two in different posts.
In these cases you need a more sophisticated approach.

While there are numerous methods for finding similar texts like [edit distance](/levenshtein) or [longest common substrings](/common-substring) they won't scale up to hundreds of thousands of ads.
If you search naively the number of possible identical pairs grows as the square of the number of ads meaning it would take weeks or months to check these 400,000 ads.
A better way is to use [MinHash](/minhash) which approximates the [Jaccard Index](/jaccard-duplicates) treating the text as a bag of n-grams.
Then you can efficiently search through these hashes using [Locality Sensitive Hashing (LSH)](/minhash-lsh), with a tradeoff of getting some false positives and false negatives.

There are a few choices you need to make with Locality Sensitive Hashing.
You need to choose your tokens and the shingle length to use.
In generally this depends on your application and the types of corruptions you would expect to see.
For job ads 3 or more words seems to have very different behaviour to 1 or 2 words; in retrospect I would pick a bigger number like 7 to reduce false positives.
In general [increasing shingle length decreases (weighted) Jaccard index](/shingle-inequality).

You can also choose how you weight your tokens.
One option is to use [TF-IDF](/duplicate-tfidf), but this had no real advantage for job ads, but it might for detecting spammers in short texts.
A big downside of TF-IDF is you need the whole set to calculate the weights, so it's much harder to run online.
I would stick to treating them as sets or multisets unless it doesn't give good enough results.

You can also normalise your tokens to increase the number of matches.
For example you may know that some sources will change case so you lowercase all your text, or remove problematic unicode characters or HTML snippets that sometimes get inserted in one text but not the other.
These kinds of changes are likely to be pretty safe, but you can spend a lot of time with them and I'd only introduce them if you actually need them.

Finally you need to choose your effective Jaccard Cutoff by selecting the number of [LSH bands and rows](/minhash-lsh).
This will depend on your application and your choice of tokens and weights.
The best thing to do is to look through a sample stratified by Jaccard (using LSH to get some higher values) and actually [look at the diffs](/python-diffs) to decide.

When you have your pairs of probably near-identical texts then you can validate them with more complex algorithms such as edit distance, now that the candidate set is low enough to run in feasible time.
You can take your duplicate pairs and add the missing links to get [sets of near-duplicate texts](/minhash-sets).
The easiest way to do this is finding connected components of the similarity graph, which works well if your threshold is high enough.
With a lower threshold you could try to group ads up using [community detection](/community-detection) or if the components are small with [cliques](/similar-companies), but if you need to it might be worth improving your validation algorithm.

I found this experience really interesting for understanding the data.
A company typically writes job ads from a template which has sections that don't change (like "about the company"), so many ads will have large paragraphs in common which will be found by Minhash.
They will also reuse an advert to advertise for a similar job, maybe in a different location or slightly more junior, so the ads are 90% the same but for a different role.
This means identifying what is a "duplicate" is quite blurry, and may depend on the application.

One technique that I didn't try is the [all-pair-binary algorithm](https://github.com/ekzhu/SetSimilaritySearch) from 
[Scaling Up All Pairs Similarity Search](/resources/scaling_up_all_pairs_similarity_search.pdf).
The paper claims to be able to find exact sets of near-duplicate items faster than MinHash.
It would be interesting to validate their claim, and see whether it holds under these conditions.

For more detail see the [Detecting duplicate job ads notebook](/notebooks/Detecting duplicate job ads.html) ([raw](https://nbviewer.org/github/EdwardJRoss/skeptric/blob/master/static/notebooks/Detecting duplicate job ads.ipynb)).
The notebook is quite long because a lot of it is looking at individual examples of job ads to gain an understanding of the data.
This is an unfortunate consequence of using the same document for exploration and presentation; it's too much text for a casual reader, but was necessary for me to understand the data.
To compensate for that I've added a table of contents with links to each section.