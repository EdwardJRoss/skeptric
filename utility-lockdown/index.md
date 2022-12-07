---
categories:
- policy
- maths
date: '2021-09-27T11:00:56+10:00'
draft: true
image: /images/covid19_cemetary_brazil.jpg
title: Utility of lockdown in an Epidemic
---

My city of Melbourne, Australia will have had 235 days of lockdown in response to COVID-19 when it is planned to slowly reopen on October 26th.
The restrictions and extended social isolation have been difficult; many people have missed important life events, some have lost their livelihoods.
However many thousands of lives have been saved (however it's not clear yet how many, as many are expected to die in the coming months).
As we wait for the city to reopen, and face the consequences of epidemic spread, I want to know has it all been worth it?
I don't have the medical expertise, philosophical expertise, psychological expertise or policy expertise to seriously answer this question - but I'll try.

# Evaluating outcomes with expected utility

To be able to say whether it has been worthwhile we need some basis for evaluating outcomes of policy interventions, and for convenience I'll use a type of utilitarianism.
The moral values used for evaluation could lead to very different conclusions; there are trade-offs between public good, liberty and fairness.
Utilitarianism, in which we optimise the expected good, is useful in that it's mathematically tractable, but not without question.
As stated in [Utilitarianism and the Pandemic](https://onlinelibrary.wiley.com/doi/full/10.1111/bioe.12771):

> Utilitarianism typically accepts that instances of goodness and badness can be aggregated in a quantitative fashion. Thus, consider a very mild pain that is caused by a physical stimulus of one unit and that lasts for 10 min. Now compare 100 instances of such a pain either spread out over 100 lives or over one life lasting many decades with a single instance of excruciating pain caused by 75 units of the physical stimulus lasting for 10 min. According to a standard utilitarian calculus the former outcome is worse than the latter, but this seems implausible. Most of us would prefer 100 instances of mild pain dispersed over our lives than 10 min of excruciating pain. It might be thought that this issue is crucial in the present context, since we will have to balance the deaths of a lower number of people against smaller burdens for a much higher number of people.

This is an oblique reference to the [Peak-End Rule](https://en.wikipedia.org/wiki/Peak%E2%80%93end_rule) where people remember the peak and end of an experience more than the aggregate value.
When we talk about utility it's different for the experiencing self from the remembering self, see Daniel Kahneman's [TED talk on Experience vs Memory](https://www.ted.com/talks/daniel_kahneman_the_riddle_of_experience_vs_memory/) for more details.
These raise real challenges with quantifying quality of life in general, and [Quality Adjusted Life Years](https://en.wikipedia.org/wiki/Quality-adjusted_life_year)in particular.
From a practical perspective it's *easier* to use the experiencing self because the experiences can be aggregated (but memories can not), but I'm not aware of good measures..

A common way to evaluate the economic benefit is to convert a life to a monetary amount and look to economic losses and spending (like in this [Crikey article](https://www.crikey.com.au/2020/10/09/covid-19-lockdown-cost/)).
The Government has an estimated [Value of a Statistical Life](https://obpr.pmc.gov.au/sites/default/files/2021-09/value-of-statistical-life-guidance-note-2020-08.pdf), currently \$222,000.
However I'm wary of any methodology based on willingness to pay; price and value are two very different things.
In some contexts this conversion may make policy sense, but there are limits to its applicability (someone can't legally pay to end another person's life).

# Costs of epidemic

An epidemic outbreak has direct deleterious effects; some proportion of people will have the severe outcome of death, some may have prolonged or permanent disability, some will have acute illness and some will have no effect.
There are also indirect effects; from worry and mourning to the overburdening of hospitals affecting everyone who works in and uses the hospital systems.
Likely a proportion of people will also self isolate.

To get a ballpark estimate of impact we should focus on the direct impacts.
For death the expected cost is the proportion of deaths multiplied by the expected (quality) years to live that were lost.
For illness the expected cost is the proportion with symptomatic illness multiplied by the expected quality of life reduction and duration.

For likelihood of death the [Economist's Excess Death's Tracker](https://www.economist.com/graphic-detail/coronavirus-excess-deaths-tracker) gives useful guidance.
Attributing cause of death is hard and from a policy perspective we're actually interested in all-cause mortality.
Excess deaths are much more reliable (although not without caveats, for example Australia's statistics exclude deaths referred to a coroner which bias the statistics), and for the COVID-19 outbreak where they are close to the deaths attributed to COVID-19 we have an indication the statistics are accurate.
Some of the hardest hit with this criterion (as of September 10th) are Peru with 1 death per 170 people, Brazil with 1 death per 350 people, Colombia with 1 death per 400 people, US with 1 in 500, and Britain with 1 death per 560 people.
As a ballpark estimate let's use 1 death per 500 people (it will depend on treatment as well as amount of spread).

People who die from COVID-19 are much more likely to be older and vulnerable of dying of other diseases such as cold and flu, and we should adjust for that.
According to [Years lost to COVID-19 in 81 Countries](https://www.nature.com/articles/s41598-021-83040-3) the average years of life lost is 16, which assuming otherwise full quality of life.
This means the cost due to death is around $16/500 \approx 1/30$.

For illness according to the [US CDC](https://www.cdc.gov/coronavirus/2019-ncov/cases-updates/burden.html) around 1/3 of people have had symptomatic COVID-19 (assuming at most one infection per person).
Assuming the average length of infection is a week, even if it reduces the quality of life to around 1/150, which is much smaller than the impact due to death.
Even assuming 10% have long COVID for 3 months this is still smaller than the impact of death (although longer term effects not yet seen could be bigger).

So the largest cost of the epidemic seems to be loss of life, which costs about 1/30th of a life year per capita.
However I feel like I'm missing something since life expectancy in Brazil and US [reduced by 1 year](https://www.nature.com/articles/s41591-021-01437-z), saying the cost should be closer to 1 year per capita.

# Cost of lockdown

The cost of lockdown are more indirect; a reduction in the quality of life due to reduction of social interactions and restrictions, as well as income lost.
However they are borne by nearly the entire population.
The expected cost is the length of the lockdown, t, multiplied by the average reduction in quality of life, q (which is necessarily between 0, for no reduction and 1).
The reduction of quality will depend on the exact kinds of restrictions that are in place, as well as mitigating policies such as payments to people who lose their jobs.
To quantify the reduction of quality of life could use things like the [Gallup World Happiness Index](https://happiness-report.s3.amazonaws.com/2021/WHR+21.pdf); there's also a less rigorous measure more directly tied to lockdown periods [using Twitter data](https://www.econstor.eu/bitstream/10419/221748/1/GLO-DP-0584.pdf) both of which give a drop in a few percent.
It currently feels worse to me, but that's probably me experiencing the end effect and [focusing illusion](https://www.purdue.edu/stepstoleaps/explore/well-being-tips/2020_0601.php).

# Direct comparison

Keeping in mind all the caveats about comparing these different kinds of quantities, let us compare them anyway.
Let's assume an effective lockdown could prevent all fatalities; how long would the lockdown need to be to be worthwhile?

Equating these the indifference point for lockdown is where 1/30 years = qt.
For a 5% reduction in quality of life under lockdown it would be worth 8 months.
However if my calculation on life is wrong given life expectancy decreased by 1 year, then it would be worth 20 *years* in lockdown.

On the other hand for a 25% reduction in quality of life it would be 1.5 months with 1/30 year per capita reduction and 4 years with a 1 year reduction in life expectancy.

So a lockdown with net positive utility probably in the range of a month to decades; admittedly too wide a range to be useful.
Tightening up the actual loss of life and the reduction in quality *may* make these estimates useful.

# Having it both ways

In reality for COVID-19 it's rare to have an epidemic without a lockdown.
Even in Peru where official fatalities are highest they still had lockdown for substantial periods of time.
The country that has come closest is Brazil, where the President Jair Bolsonaro opposed lockdown and other preventative measures, there were still [city level restrictions](https://www.npr.org/sections/goatsandsoda/2021/03/23/980391847/brazil-is-looking-like-the-worst-place-on-earth-for-covid-19).
All the other countries had significant lockdowns to protect their health care systems.

The early lockdowns in Melbourne, and the lockdowns in New Zealand have been unusual in that they have greatly reduced epidemic spread (for example [Victoria had negative excess mortality](https://www.abs.gov.au/articles/measuring-excess-mortality-victoria-during-covid-19-pandemic) after the first outbreak).
But now spread in Melbourne and Sydney is forcing Australia to face epidemic spread as vaccinations are rolled out.
Australia has been [slow in the vaccine roll out](https://www.bbc.com/news/world-australia-56825920) relying primarily on AstraZeneca's vaccine which due to a rare blood clotting side effect which led to ATAGI [recommending it only for over 60s](https://www.health.gov.au/news/atagi-statement-on-revised-recommendations-on-the-use-of-covid-19-vaccine-astrazeneca-17-june-2021) based on [risk ratios](https://www.health.gov.au/sites/default/files/documents/2021/06/covid-19-vaccination-weighing-up-the-potential-benefits-against-risk-of-harm-from-covid-19-vaccine-astrazeneca_2.pdf) assuming Australia would secure more Pfizer vaccine before any serious epidemic outbreaks (which seems to have been largely correct outside of Melbourne and New South Wales).
However at least vaccination of the elderly and protection of aged care facilities this current outbreak in New South Wales has half the fatalities despite double the cases of Melbourne's outbreak (which won't be significantly impacted by a 14 day lag).

Reportedly the vaccines provide a roughly 90% reduction in hospitalisation, a 95% reduction in death and a moderate reduction in infection.
It's hard to compare statistics here because of different stages of roll-out and different strain, but an effective lockdown program could reduce the number of deaths by up to a factor of 20 (and more given the reduced load on hospitals).

# Melbourne's next steps

Right now feels very hard in Melbourne and Sydney.
Each city has a roadmap to slowly unwind restrictions based on vaccination targets which are limited by supply of Pfizer and Moderna vaccines (due to AstraZeneca vaccine contraindications in younger people) and to a lesser extend the interval between doses.
It's hard watching this when Europe and the US have been more open for longer, since the Federal Government has been too slow with the supply of the mRNA vaccines.
But Europe and the US have had *much* higher fatalities and case numbers, on top of significant restrictions and periods of lockdown (dependent on location).
It's easy to forget earlier in the year when Melbourne was relatively open and those countries were much more closed with deaths piling up.

Melbourne's roll out plan is based on [modeling from the Burnet Institute](https://burnet.edu.au/system/asset/file/4929/Burnet_Institute_VIC_Roadmap_20210918_-FINAL.pdf) which was run on the [open source COVASIM](https://github.com/InstituteforDiseaseModeling/covasim).
The level of transparency and scientifically informed policy making is commendable.
The Burnet modelling does not include a "fully open" scenario (which would have been an easy way to score cheap political points), but is relatively clear.

However it does seem somewhat conservative in predicting 1455-3142 Victorian deaths in the period July-December 2021, that is 1 death per 1200-2800.
Assume very conservatively that we have an unvaccinated death rate of 1 in 550, that of the UK across the *whole* pandemic, and assuming 70% population are vaccinated with a 80% reduction in death then we would have a death rate of 1 per 550/(1-0.7*0.8)=1250, that is 3200 deaths.
This seems quite unlikely; even if that's the long term outcome over the next year I'd be surprised if it's squashed into the next few months.
However they do seem plausible for the long term impact (although we already have 800 deaths from the first wave I would discount), with the same rate of 1 per 550 if we get to 85% vaccination with 95% efficacy we get closer to the 1 in 2800 mark.

For our remembered self a Freedom Day would be a very planet way to end the pandemic.
But it could be catastrophic for healthcare dealing with all the cases at once, especially since vaccine uptake is likely to be continuing.
I wonder given the most vulnerable are already vaccinated whether there could be more opening up sooner, but in a large population the statistics really matter.
Exponential growth could overwhelm hospital capacity (which would be managed, but at a cost to staff and patient welfare), and even a 0.1% saving of lives in a population of 4 million is 4000 lives.
I'm not totally convinced that, ignoring our sunk cost, on utilitarian grounds how much longer we should wait to open up, but the reduction of suffering (not to mention fairness) of giving everyone an opportunity to get vaccinated is large.
And so we wait.