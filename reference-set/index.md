---
categories:
- data
date: '2021-03-08T20:50:43+11:00'
image: /images/reference_set_venn.png
title: Reference Sets as Pervasive Models
---

Suppose you have a long standing heart condition, and are considering undergoing a surgical procedure that could alleviate the procedure, but has its own set of risks.
You happen to have a good friend who's a statistician at the hospital you're being seen at and can get you historical frequencies of complications.
However she asks you what reference set do you want to use?

Do you want complications related to just the specific procedure treating your condition, or for all procedures on that area of the heart for a variety of conditions?
And do you want the complications for all procedures worldwide, or just in your country, or perhaps just in this hospital, or under this surgeon?
Do you want to segment it by sex, age group and BMI category?

The more specific you make the comparison set the more related the data is to your specific outcome, but the less data you have left.
Just looking at all surgeries from this particular surgeon on this particular condition for someone with a similar medical history like you may only leave 10 or 20 examples, and you can't judge the probabilities of a serious complications that happen only in 3% of cases.
On the other hand including surgeries for people much older or less healthy than you, or from hospitals using older, and more risky, procedures could make the incidence of bad outcomes appear more likely.

In the end you need to use judgement in picking a good reference set.
You should identify the factors that are most strongly associated with a poor outcome, and only filter on those.
That requires an understanding of the underlying mechanisms of the risk.

In the end this is just a [constant model](/constant-models) for classification, where we're trying to pick the cases to include carefully.
If you want to use the information more efficiently you need to build a model on the data.
What does the incidence of complications on elderly people tell you about the likelihood on younger people; it almost certainly tells you *something*, but excluding it treats it as if it tells you nothing.