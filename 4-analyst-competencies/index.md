---
categories:
- data
date: '2020-05-02T22:01:02+10:00'
image: /images/kolassa-data-scientist-venn.png
title: Four Competencies of an Effective Analyst
---

Analysts tend to be natural problem solvers, good at reasoning and adept with numbers.
But to know how to frame the problem and what to look for they need to understand the context.
To solve the problems they have to collect the right data and perform any necessary calculations.
To have impact they need to be able to understand what's valuable, communicate their insights and influence decisions.

These make up the four competencies of an effective analyst.

* Problem solving, logical thinking, reasoning and statistics
* Subject matter expertise and business acumen
* Adept with technology, information systems and programming
* Communication, stakeholder management and influence

I really like the 4-way Venn diagram of [Stephen Kolassa](https://datascience.stackexchange.com/questions/2403/data-science-without-knowledge-of-a-specific-topic-is-it-worth-pursuing-as-a-ca/2406#2406) for illustrating this; he calls it a "The Perfect Data Scientist" but it's getting to the same point.

![Data Scientist Requires Communication, Statistics, Programming and Business](/images/kolassa-data-scientist-venn.png)

It's very difficult to get someone adept at all of these things, and when building an analytics team it will be most effective with people that have complementary skills collaborating.
For an analytics professional it makes sense to build out some strength in each of these areas.

# Quantitative problem solving

The bread and butter of analytics is understanding what impact actions have on metrics of interest.
This requires having a model of the system, knowing the interactions and understanding measurement.
The models can range from informal sketches to fully specified statistical models.
Interactions in the measurement can be managed by understanding the order of magnitude of effects or through statistical testing or causality modelling.

Sometimes these techniques can be overemphasised and the focus can be more on the cool new modelling technique (say, deep reinforcement learning) than how effectively it solves the problem.
In practice a core of simple techniques can be effective and efficient for a wide range of problems.
Sometimes for highly valuable problems more advanced techniques can pay huge dividends, but they are often time consuming to implement, harder to communicate and maintain and don't always work out.

# Subject Matter Expertise

Some of the best analysts are the people who have a wealth of experience in their area.
This is a very specific kind of skill, subject expertise in digital marketing won't help you in novel drug discovery.
This is part of what makes it valuable; the only way to get it is to spend a long time working in the area.

For problem solving you need at least an informal model of the system, understanding how measurements are collected and being familiar with issues that will occur.
These come naturally with deep experience in the area; understanding deeply how things work (and how to differentiate what's important from what isn't) and having been exposed to what goes wrong.

To be able to communicate and influence it's important to speak the same language as your stakeholders and know what matters to them.
Subject expertise helps immensely because you know the industry terminology and what people worry about.
This makes it easier to build trust and communicate insights.

Any business will have people that are experts in the area you're analysing.
The best way to gain expertise is to talk with them frequently to understand whether your models make sense and that you can be understood.

# Technology

Technology enables complex analyses, handling large or diverse datasets, automating workflows and delivering bespoke solutions.
At minimum you need to be able to access and analyse relevant data, and even when it's collected manually this normally requires some technology.
Generally there's some amount of data cleaning, integrating different datasets and performing calculations and creating visualisations.

Specific technologies are often disproportionately represented in job ads for analysts.
Typically you can learn to use different systems, especially ones that built specifically for analysts.
Some are specific to problem solving techniques, like [Stan](https://mc-stan.org/), or to a particular domain, like Google Analytics or Adobe Analytics to web and mobile applications.
When communicating with other people you may be better off using technologies that are familiar to them; [Excel](/using-excel) is popular.

Building bespoke technology systems can automate time-consuming processes and make whole new things possible.
However maintaining these systems is often expensive and requires continual investment.
In the worst case you could become a full time developer operating and maintaining a software system.

Technology makes new things possible, but it's often best to use existing specific tools or leveraging other technical expertise than building your own solutions.
However even using these tools and interacting with technical experts requires being adept with technology and being able to think about systems.

# Communication and influence

Analytics is using quantitative information to inform which decision will best deliver desired outcomes.
Even if you have the most rigorous and specific analyses, if you can't persuade the decision maker to act based on them they are worthless.
To understand what problems to solve you need to understand what options the decision maker has and what outcomes are desired.

Communication can help guide how to solve problems.
I've sometimes come up with a complex metric or analysis that fits the problem well, but is very hard to communicate with my stakeholders.
Often there's a simpler framing that results in the same decision but is easier to understand.
Being easy to understand is highly valuable; it will be much clearer if it makes sense.
A decision is often made based on a number of factors outside the analysis, and the decision maker needs to understand how the analysis fits in the bigger picture.

Communicating with data is a particular skill set.
Knowing how to present data and emphasise what matters to your stakeholders can be all the difference in influencing a decision.
I really like [Cole Nussbaumer Knaflic's Storytelling with Data](https://www.storytellingwithdata.com/) as a starting point here.

Even in a more product delivery role you will still need to persuade your users to follow the recommendation of your system, and persuade people in your company that your system adds value.
In both of these cases understanding their objectives and being able to clearly communicate what value you're delivering will make it much more effective.

# Bringing it all together

These competencies are very broad and you could spend a lifetime learning any one of them.
Whether you're [building systems or influencing decisions](https://towardsdatascience.com/ode-to-the-type-a-data-scientist-78d11456019) understanding how all these pieces fit in your objectives, and how to mitigate the deficiencies is important.
You can always work with experts in statistics, business, technology or communication to get things done.
But you'll have to have enough overlap to understand what actually needs to be done and convey why it is important.