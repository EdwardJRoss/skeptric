---
categories:
- insight
date: '2020-10-11T21:30:31+11:00'
image: /images/australian_population_age.png
title: Australian Deathographics
---

I've recently tried to [estimate Australian Deaths](/australian-deaths) using life expectancy.
This failed badly and I think the reason is demographics; this article looks more into this.

The [Australian Bureau of Statistics has population by age](https://www.abs.gov.au/statistics/people/population/national-state-and-territory-population/latest-release), and the Australian Institute of Health and Welfare have [Mortality Over Time and Regions (MORT)](https://www.aihw.gov.au/reports/life-expectancy-death/mort-books/contents/mort-books) which summarises the current probability of death by age range.
Here is a super summarised version of this data:

| Age   | Population | Death Rate | Population Deaths | Fraction of Deaths |
|-------|------------|------------|-------------------|--------------------|
| 0-19  | 25%        | 0%         | 0%                | 0%                 |
| 20-39 | 29%        | 0%         | 0%                | 0%                 |
| 40-59 | 25%        | 0%         | 0%                | 0%                 |
| 60-79 | 17%        | 1.5%       | 0.25%             | 42%                |
| 80-84 | 2%         | 5%         | 0.1%              | 17%                |
| 85+   | 2%         | 13%        | 0.25%             | 42%                |

Notice that population deaths is the product of the previous two columns and fraction of deaths is the normalised population deaths.
The life expectancy is the sum of the product of fraction of deaths with age; it's hard to estimate because of the 85+ band but it's likely in the 80-84 range.
It's hard to estimate the average age of death from bands (life expectancy), it's the product of 
The median age of death is in the range 80-84 as well.
These are consistent with an Australian life expectancy of 82 years.

Multiplying and summing population with death rate gives an overall probability of death of 0.6%.

# Understanding the errors

So why did a rate of 1/life expectancy, give 1.2% and not 0.6%?
And why are [300,000 babies born](/australian-birth-check) and the total fertility rate is less than 2 per woman, yet there are only 150,000 deaths so it seems like the population is increasing?

It comes down to demographics; if we assume a constant birth rate and use the current survival rates we get a very different picture of population.
The 80-84 age band and 85+ age band are both 4%, about twice as big as we actually see, and the 60-79 age band is 20% of the population, a little bigger than the 17% we see.
This would get us closer to a death rate of 1% in the ballpark of 1/life expectancy.

So why are there so few people above 80?
It doesn't quite make sense to be World War II since that started about 81 years ago, and couldn't account for fewer people 82 and older.
Another possibility is that the survival rates have changed a lot in the last couple of decades and so we have fewer over 80 year olds.
Yet another is we have more migrants in the 20-39 age range outpacing older age brackets.
I don't know if any of these explanations hold water, or if it is something else entirely.

I think I understand now why I estimated the death rate so poorly, but I still don't have a simple model I could take to other countries to estimate their deaths.
This would require better lumping and ways of estimating elderly populations (relative to age of death in that place and time).