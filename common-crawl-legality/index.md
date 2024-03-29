---
categories:
- general
date: '2020-08-25T08:00:00+10:00'
image: /images/common_crawl_terms_of_use.png
title: Legality of Publishing Web Crawls
---

As a data analyst I rely on open code and open data to inform decisions.
There's a lot of data available on the web which would be great to transform and make openly available to the community.
However it's not my data to give, and I'm concerned whether it would violate copyright.

An interesting aspect is there *are* companies that scrape data from all over the web to use for analysis.
A few of them are Google, Microsoft, Apple and Amazon.
The scraped data is used for keyword analysis and powers their search products, among other things.
They don't verify terms and conditions of every website they visit, at most checking the `robots.txt` to see what parts of the site they've been able to index.
However they don't share this data.

[Common Crawl](https://commoncrawl.org) is an organisation that does the same thing as the tech giants (although on a smaller scale), but releases their data in the open (the [Internet Archive Wayback Machine](https://archive.org/web) is another example).
The scraped websites [HTML and metadata is available](/text-meta-data-commoncrawl), they have a useful [columnar index](/common-crawl-index-athena) of the pages they've captured, and I've used it to build a reasonable [sample of job ads](/common-crawl-job-ads).
This dataset has been immensely beneficial in Natural Language Processing, and is often used in Information Extraction and large Language Models; for example it's a large part of [RoBERTa](https://arxiv.org/abs/1907.11692).
Larger datasets would be even more useful; OpenAI curated their own web crawl for the powerful [GPT-2](https://openai.com/blog/better-language-models/).

How do they do this without breaching copyright?
An [excellent article from Forbes](https://www.forbes.com/sites/kalevleetaru/2017/09/28/common-crawl-and-unlocking-web-archives-for-research/) covers some of this from the director Sara Crouse.
She raises three main points:

1. Many legal advisers are willing to work with them on Copyright, some of them Pro Bono
2. The web pages are not available for easy consumption
3. The crawls are only a sample of the available pages on a given domain

These points are useful in considering whether you should publish your [own web crawl](/request-warc) data.

## Availability of Lawyers

The reality of laws is that they need to be enforced.
Copyright infringement is only becomes a practical issue when you receive a letter from a lawyer.
If you need to defend yourself in court, even if you're in the right, it could be a huge cost in legal fees.

Common Crawl is a non-profit organisation trying to archive the web.
They've got a bit of money and a lot of goodwill.
If they get sued the litigator won't always come off in a good light, and there are lots of lawyers willing to work with them to test the law on copyright.

If you're an individual releasing some data scraped from a company's website for research purposes, they are likely to have more at stake and more resources than you.
For a small project you're not likely to get a Pro Bono lawyer and would have to pay the legal costs yourself.
If they decide to pursue legal action the time and cost would not be worth it for you, and it would likely be wise to take down the material if asked to do so.

If the data is user submitted content (for example a [Reddit Dataset](https://archive.org/details/2015_reddit_comments_corpus)) the actual owner of copyright will typically be the users.
This means, depending on the terms of the publishing site, they may not own the copyright and may not be able to legally pursue you because of it.
But I don't know whether that's true, and the only way to find out is to pay some legal professionals for advice, and maybe even resolve it in court.

So the pragmatic answer to can I publish this is, will the source of the data be interested in legally pursuing you over it?
If you're putting their business model at risk the answer is almost certainly yes.

## Ease of Access

I suspect the argument that the data is difficult to consume wouldn't hold up in court.
The difficulty to access a resource doesn't seem to have any bearing on whether you have the right to republish that information.
Besides it not as difficult as they make it sound in the article; a software engineer could make it easy with some straightforward work.

However because it's not aimed at human consumption it doesn't touch the business model for many websites; attention and advertising revenue.
This means they have little reason to pursue legal action; many website owners are probably not even aware their data is in Common Crawl.

If a published dataset has low visibility and low impact on the business then in practice it may be harmless to publish.

Ideally you would contact the websites you are sourcing the data from directly and ask them the right to publish the data.
But in general unless there is a large benefit to them they are likely to decline your request and potentially order you to remove the material.

## Fair Use

The most interesting part of the argument for Common Crawl is that they just have a small sample of web pages from each domain.
This resonates with the idea of Fair Use where reproducing a small number of pages from a book is reasonable (which is how Google Books exists).

This won't be true for a lot of data analysis crawls.
Typically you want structured data, which requires targeting a few websites in depth.

In any case you would need lawyers to resolve whether it is fair use, but if you're only taking a small sample of the site it's more likely to be viewed that way.

## Ethics

A separate question to "can I share this dataset" is "should I share this dataset"?
I would say that if it is from a publicly available datasource and won't lead to significant harms then it is a reasonable thing to do.

I consider data published on the open web (that is not behind a paywall or login) to be freely available.
This doesn't mean you can claim the content is your own and modify it.
But it seems strange to say you can't keep a copy of that web page and share it, since it was at one time made available to everyone.

Of course you don't want to do any harm because of it.

Public Transport Victoria [publicly released data](https://www.theage.com.au/national/victoria/you-can-do-what-you-like-with-the-data-personal-myki-details-exposed-20190815-p52hdc.html) that can be used to identify individual commuters personal information.
Even though this was a public release I would carefully consider republishing it because it could do harm to individuals, and is likely to bring marginal benefits.
The [NSW Opal Dataset](https://opendata.transport.nsw.gov.au/dataset/opal-tap-on-and-tap-off), while less interesting as it is in aggregate, is still useful while likely to do minimal harm.
It would be great to see this sort of data released when it is unlikely to do harm, say in 50 years.

Similarly if I extracted a large amount of information I wouldn't want to release email address as a column.
Even if emails were in the original data, having done the work of extracting them it makes for an easy target for spammers.
It seems unlikely there's much benefit for publishing email addresses, so better to redact them.

These decisions are rarely clear cut; but even if it is legal it is worth considering whether republishing data will bring any harm.