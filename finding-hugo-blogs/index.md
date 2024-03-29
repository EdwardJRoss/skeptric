---
categories:
- sql
- blog
date: '2020-12-07T23:27:49+11:00'
image: /images/hugo_bigquery.png
title: Finding Hugo Blogs with BigQuery
---

I want to find examples of other Hugo blogs, but they're not really easy to search for.
Unless someone put "Hugo" in the descrption (which is actually common) there's no real defining files.
However there's a set of files that are in a lot of Hugo Blogs and we can search them in Github with the [GHArcive BigQuery Export](https://www.gharchive.org/#bigquery).

The strategy is that most Hugo blogs will contain a `/themes` folder, a `/content` folder a `/static` folder and a `config.toml` file.
They don't *have* to have these, but many of them will as these are very standard structure.
We can then search for them in Bigquery (this is just a 1% sample, to get them all replace `sample_files` with `files`):

```
select repo_name, markdown_files from (
SELECT repo_name,
      logical_or(REGEXP_CONTAINS(path, '^themes/')) as has_theme,
      logical_or(REGEXP_CONTAINS(path, '^content/')) as has_content,
      logical_or(REGEXP_CONTAINS(path, '^static/')) as has_static,
      logical_or(path = 'config.toml') as has_config,
      count(case when ends_with(path, '.md') then 1 end) as markdown_files
FROM `bigquery-public-data.github_repos.sample_files`
group by 1
)
where has_theme and has_content and has_static and has_config
limit 100
```

The results aren't all blogs, some are examples like [hugo-lightslider](https://github.com/pcdummy/hugo-lightslider-example) or [hugo-deploy](https://github.com/nathany/hugo-deploy).
But they are all related to Hugo in some way.

This could then be used for further queries, like finding the most popular themes, the distribution of number of posts, extracting tags from blog posts for a classifier and other things.
