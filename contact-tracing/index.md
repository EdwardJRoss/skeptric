---
categories:
- data
- maths
date: '2020-09-07T20:26:25+10:00'
image: /images/contact_tracing.png
title: Contact Tracing in Fighting Epidemics
---

The state government of Victoria, Australia has [recently announced](https://www.theguardian.com/australia-news/2020/sep/06/melbourne-stage-4-coronavirus-lockdown-extended-for-two-weeks) a plan on how to respond to the current Covid-19 pandemic.
Based on epidemiological modelling they have set to reduce restrictions based on 14 day averages of new case numbers.
If the 14 day average daily new cases are 30-50 in 3 weeks they will reduce restrictions; if they are below 5 a month after that they will reduce restrictions again.

However through the lens of the [SIR model](/sir) this seems surprising.
The rate of epidemiological spread is independent of the number of people that have the disease.
The key factor I overlooked is the stochastic nature of the model at an individual level, and the impact of voluntary isolation and contact tracing.

# Individuals in the SIR Model

In the SIR model an infectious individual on average infects $R_0$ other people while infected.
In a population where most people are susceptible if this number is greater than 1 then the epidemic spreads, if it's less than 1 then the disease will die out.

One simple method for reducing the number of other individuals infected is to isolate them.
If infectious people don't interact with anyone else then they can't infect anyone else.
If everyone does this then the disease dissapates very quickly.

The problem is that a lot of people don't *know* that they're infectious.
With many diseases people are infectious before they show symptoms, and some people don't ever show symptoms but are infectious.
So they go on as before and infect many other people.

A really simple model for this is that asymptomatic carriers infect a high number of other people $R_h \gg 1$ and symptomatic cariers infect a low number of people $R_l \ll 1$.
Then if the chance of being asymptomatic is $p$ the overall $R_0 = p R_h + (1 - p) R_l  \approx p R_h$.

# Contact tracing

Many infectious diseases spread mainly through direct contact between individuals.
If someone has tested positive then their direct contacts are much more likely to have it than random individuals.
If you test those individuals immediately you can effectively detect asymptomatic carriers and get them to isolate.

This then moves these asymptomatic carriers from the $R_h$ group into the $R_l$ group, effectively reducing $p$.
If you can get $p$ low enough then the epidemic spread will stop.

However contact tracing is a lot of work.
Every person tested positive need to be contacted and interviewed; this could take some time if they are hard to get hold of.

Then you have to follow up with each of their contacts and the information of where they had been.
If they'd been to work or for a haircut you would need to call the business and get the contact information of everyone that was in contact with that individual.
This may take some time and quickly turn out to be 20 or 30 people that need to be contacted.

I think it would be reasonable to estimate that every new infected case would maybe 1-2 people days to follow up on average.
This may sound high but getting in contact with people can be hard.
It's also important how quickly you get into contact with them; the longer you leave it the more people they have infected.

The [NSW Health Website](https://www.health.nsw.gov.au/news/Pages/20200416_01.aspx) claims their team of 150 can make up to 1300 calls per day.
So if each positive person is on average in contact with 10 other people they could handle up to around 120 new positive cases per day.

So if you have around 500 new cases every day, then you'd need of the order of a thousand case workers plus relevant support.
The Victorian Government [claimed](https://www.theaustralian.com.au/breaking-news/coronavirus-victoria-claims-states-tracing-team-half-the-size-of-nsw/news-story/53f91f53024d9f1fcd4f5b76558f1c0a) it had a team of 2600, but couldn't keep up with 500 per day.

I'm not sure what the difference is here (probably who they include in the counts), but for the sake of argument let's say a manageable caseload for Melbourne or Sydney is up to around 150.

# Low targets

Things don't still add up.
If contact tracing is the bottleneck then why would we be aiming to get to only 5 daily cases and not 150?
This could be reasonable handled by a small team for a city of 5 million.

I think the reason is stochastic outbreaks.
You're sometimes going to get unreasonable people who refuse to be interviewed and refuse to isolate.
They could infect a lot of other individuals, some of whom will become asymptomatic carriers.

Due to chance sometimes you will get a few of these individuals on the same day.
They will infect a bunch of people, and even the symptomatic individuals won't show up for a week.
Because of the exponential growth of epidemic disease by the time you get the first positive test you could have scores of people infected.
In the worst case you could end up a few weeks later with a couple hundred cases per day and be unable to trace them.
The more cases you miss the faster the infection spreads.

The chance of this happening will be proportional to the number of active cases in the community.
In turn this is roughly the total number of new cases in the past couple weeks (if we assume the average time in the infectious state is two weeks).
So you would set targets based on the number of new cases in the past couple of weeks.
Where you set this target depends on how much risk you're willing to take, and how many unreasonable people you think there are.

# Reflecting on the model

This analysis goes past our initial simple SIR model, but not incredibly far.
One big step forward would be to model individuals with a stochastic rate of recovery and rate of transmission.
You would have to choose distributions for these and fit them to the data, but I'm sure there are standard models.
Then your "unreasonable people" will be the tails of the distribution.

However making these sorts of decisions on a stochastic model is painful.
How big a chance are you willing to take on another "wave" of spread?
Is 5% too much? What about 0.5%?
The costs of restrictions on businesses and mental health of individuals is significant.

The outcomes of the model are also going to be very sensitive to the assumptions built in.
It would be hard to get reliable data for a lot of parameters because of missing information of individuals who don't get tested.
Using epidemiological information from other outbreaks would be hard because it's hard to untangle all the differences between two cities and the impact that has on spread.
I wouldn't be surprised if there's a factor of 4 difference between the probabilities at the tails of two different modelling groups.

The Australian Prime Minister Scott Morrison has [criticised Victoria's contact tracing](https://www.smh.com.au/politics/federal/morrison-pressures-victoria-to-lift-its-game-on-contact-tracing-20200907-p55tar.html).
He uses the example of New South Wales which sustained daily positive tests between 10 and 20 for the last two months.
However single examples aren't great for statistical modelling; they could easily have an outbreak next month.

More significantly leading epidemiologists think that the [targets are conservative](https://www.theage.com.au/national/victoria/epidemiologists-react-to-victoria-s-road-map-out-of-stage-four-lockdown-20200907-p55tam.html).
These are people who actually know what they're talking about (unlike me); so I'd guess that the decision was made on a very conservative chance of a second wave of spread with many bad case assumptions.