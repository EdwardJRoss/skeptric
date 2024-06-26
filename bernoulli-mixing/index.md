---
categories:
- maths
date: '2020-04-27T08:28:18+10:00'
image: /images/mixture_binomial.png
title: A Mixture of Bernoullis is Bernoulli
---

Suppose you are analysing email conversion through rates.
People either follow the call to action or they don't, so it's a [Bernoulli Distribution](https://en.wikipedia.org/wiki/Bernoulli_distribution) with probability the actual probability a random person will the email.
But in actuality your email list will be made up of different groups; for example people who have just signed up to the list may be more likely to click through than people who have been on it for a long time.
But we can ignore this distinction and treat them as a single group, when it makes sense to, for experiments.

Concretely suppose that 10% of our email list is new members and 90% is existing members.
New members have an 80% conversion rate, and existing members have a 40% conversion rate.
Intuitively the overall conversion rate is $0.1 \times 0.8 + 0.9 \times 0.4$ or 44%.

This situation is have a [mixture model](https://en.wikipedia.org/wiki/Mixture_model).
When we send an email we can think of the group we send it to being a draw from a Bernoulli distribution, which then determines which Bernoulli distribution we sample from.
Formally we pick our groups with a random variable $G \in \mathrm{Bernoulli}(\mu)$ which picks between $X \in \mathrm{Bernoulli}(p_1)$ and $Y \in \mathrm{Bernoulli}(p_2)$.
In our example $\mu = 0.1,\ p_1 = 0.8,\ p_2 = 0.4$.
The overall distribution is then $Z = G X + (1 - G) Y$.

Enumerating the possible values of the variables and their likelihood gives $Z = 1$ if and only if $G = 1,\ X = 1$ or $G = 0,\ Y = 1$, otherwise it is zero.
The probability of this is $\mu p_1 + (1 - \mu) p_2$, which is exactly the intuitive calculation we used above.
So we see that $Z \in \mathrm{Bernoulli}(\mu p_1 + (1 - \mu) p_2)$.

As you'd expect you can ignore the subgroups and treat the combined group together.
This would make sense in an experiment where you expect similar behaviour accross groups, or you're just interested in the overall effect.
Because there's only one response variable it's not possible to distinguish the groups by the conversion rate alone.
Note that this effect isn't true for other kinds of models, a mixture of binomials is generally different to a binomial.