---
categories:
- maths
- data
date: '2020-09-06T20:06:41+10:00'
image: /images/epidemic.png
title: Modelling the Spread of Infectious Disease
---

Understanding the spread of infectious disease is very important for policies around public health.
Whether it's the seasonal flu, HIV or a novel pandemic the health implications of infectious diseases can be huge.
A change in decision can mean saving thousands of lives and relieving massive suffering and related economic productivity losses.
The SIR model is a model that is simple, but captures the underlying dynamics of how quickly infectious diseases spread.
They can be used to understand many phenomena we see with the spread of real diseases.

# A rough model of infectious disease

To understand why the model works we need a basic understanding of infectious diseases.
I don't know much about infectious diseases but here's the rough model I have in my head.
Reality is much more complicated and nuanced; but this is good enough to explain some things.

An infectious disease is caused by pathogenic microorganisms, such as a virus or bacteria.
These organisms replicate and multiply inside an infected host.
The organisms can then be transmitted to other individuals and infect them.
The way they spread depends on the disease; HIV requires direct exchange of bodily fluids, but you can get influenza from someone coughing near you as infected droplets get into your mouth, nose or eyes.

When someone gets an infectious disease typically they have an immune response that fights of the infection.
The human body has millions of different types of antibodies that can each lock onto a specific molecule of a foreign substance, like a lock and key.
When they bind to that molecule they start a process of destroying that molecule, and send signals to create more copies of themselves.

As more pathogen replicates in the body it's more and more likely to run into a matching antibody that is the lock to the key of a molecule on the surface of the pathogen.
The antibodies start multiplying and destroying the invaders.
The immune system, now recognising a threat, does whatever it can to drive them out.
In the best case they recover completely, in the worst they can die.

After someone has had an antibody response many of the antibodies produced will go away.
However there will be many more of that specific type of antibody than there were before.
If the same type of pathogen comes back into the body then it will be found much more quickly because there's a lot more of that antibody out looking for it.
Consequently recovery will be much faster and the individual may not even become infectious.

Immunisation works by boosting this kind of antibody without exposure to a substance that replicates quickly.
This is often done by trying to craft a non-replicating version of the pathogen with the same outer layer for antigens to bind to.

Incidentally allergies are when the antibodies start binding to a benign substance, such as pollen.

# SIR Model

The SIR model is a family of mathematical models for representing the spread of disease through a population.
The model is named after the three stages an individual can be in; susceptible, infectious and removed.
A susceptible can contract the disease from an infected individual.
An infectious has the disease and can spread it to susceptibles.
A removed can't become infectious; they are immune to the disease.

A proportion of the population s is susceptible, i is infectious and r is removed.
Since these are the only possibilities s + i + r = 1.
There is only one path through the states: start off susceptible, then become infectious and finally are removed.

An individual goes from infectious to removed either by their immune response destroying the disease, or by dying.
It's a bit morbid but in the model we treat "recovered" and "dead" as the same.
Obviously dead is a much worse outcome than recovered, but just for understanding the dynamics of the disease they're the same - they won't get sick again.
After understanding the spread you would then look into modelling mortality.

The average time it takes for someone to go from infectious to recovered is t, the recovery time.
Then each day the number of recovered individuals is i/t, since after t days you would expect each of them to be recovered.

For a susceptible person to become infected they need to be in contact with an infected individual.
Suppose that on average an infected person contacts b people at random each day.
Then of those b people, the proportion of them that are susceptible is s, and they become infectious.
So the proportion of susceptible people who are infected each day is b ✕ i ✕ s (each infectious person contacts b other people, of which the fraction s are susceptible).

This model is of course very simple.
People don't go running around contacting people at random; they tend to gather in cliques such as neighbours, co-commuters, work colleagues and family.
And when they get the disease typically there's an incubation period before they become infectious; it doesn't happen immediately.
You also can't have a fraction of a person infected.
But it's in the right ballpark, and will give roughly the right form of the dynamics without trying to model all these extra complicated factors.

In summary we have three equations about how much each proportion changes each day:

* $\Delta s = - bis$
* $\Delta i = bis - i/t$
* $\Delta r = i/t$

Or to put it in a different notation if we have at day n, $s_n$ susceptible, $i_n$ infectious and $r_n$ removed then:

* $s_{n+1} = s_n - b s_n$
* $i_{n+1} = i_n + b i_n s_n - i_n / t$
* $r_{n+1} = r_n + i_n / t$

These are simple equations that you can put into a programming language or a spreadsheet.
All you need is the initial proportion of susceptible, infectious and recovered and estimates of b and t.

# Analysis

The benefit of having a simple mathematical model is we can analyse it without implementing it to understand the dynamics.
There's a lot that we can understand from these equations.

The first thing to observe is that all the equations are in terms of the fraction of population.
That means that the total size of the population doesn't really matter at all; the spread through a town of 10,000 people is the same as the spread through a city of 10 million as long as b and t are the same.

## Change in susceptibles

The number of susceptible people is decreasing.
This is because all a susceptible person can do is become infectious.
When there are no infectious people left then the number of susceptible people doesn't change; which makes sense since there's no way they can get infected (obviously the initial source of infection is beyond this model!).

Also note the more susceptible people there are, the faster they decrease.
If the number of infectious people is constant then susceptibles drop off exponentially like $e^{-bi}$.
If the number of infectious people is growing then it will drop off even faster.
This is why infectious diseases are so scary; they can spread exponentially through a population.

## Change in infectious

The rate at which the number of infectious people changes is really crucial; this is how quickly people are getting sick.
It's useful to rewrite the equation $\Delta i = bis - \frac{i}{t}$, as $\Delta i = (bts - 1) \frac{i}{t}$.
If the product b ✕ t ✕ s is greater than 1 then the number of infectious people is increasing.
If it's less than 1 then the number of infectious people is decreasing.

The product b ✕ t is called the basic reproduction number $R_0$.
Recall that b is the number of people an infectious person will come into infectious contact with per day.
And t is the average number of days a person will be infectious.
So their product is the average number of people an infectious person will infect.

When $R_0 s \ll 1$ then $\Delta i \approx - \frac{i}{t}$ and the number of infectious people drops off exponentially like $e^{-1/t}$.
In this case the infection dies out, and approximately $s e^{-i R_0}$ more of the population becomes infected.

When $R_0 s \gg 1$ then the number of infectious people grows exponentially like $e^{R_0 s}$.
Since s is decreasing this rate of increase will slow down.

The critical point is when $R_0 s = 1$, or equivalently $R_0 = \frac{1}{s}$.
Here the rate of change of infectious is zero; we've hit the maximum of infectious cases.

The maximum number of infectious cases is very relevant for hospitals.
Some fraction of infectious individuals will need to be hospitalised or they could have severe repercussions or even die.
So to minimise harm hospitals should have enough staff and beds to cover this fraction of the infectious individuals at peak.

So if initially $R_0 s \ll 1$ then the infection will die out quickly.
However if $R_0 s > 1$ then the infection will grow exponentially (and susceptibles will decrease exponentially) until $R_0 s = 1$ and then it will start to die out.
In the second case we have an epidemic and a large fraction of the population will have contracted the disease.

## Basic Reproduction Number

It's clear that the dynamics are really strongly governed by the product $R_0 s$.
For a new kind of disease, s will be close to 1; almost everyone is susceptible.
Then all you can do is to reduce the basic reproduction number.

Reducing the basic reproduction number is all about reducing the number of infectious contacts.
The simplest method is reducing the number of people contacted; quarantining infected individuals so that they can't come into contact with susceptibles.
Another tactic would be to try to isolate subpopulations (each of which has an independent SIR model), to try to contain spread between groups.
Generally methods to decrease the density and variety of contacts within a population will slow the spread.

Another method is to reduce the likelihood of an infection on contact; for example using physical barriers such as masks and gloves for airborne diseases and condoms for sexually transmitted diseases.
Many diseases can spread indirectly through contact with a surface, so regular cleaning is important too.
Similarly any treatment to reduce the infectious period, or reduce the amount of pathogens excreted would help reduce this number.

It's notable that it's the product $R_0 s$ that matters.
If the basic reproduction number is 4, but much less than a quarter of the population is susceptible then the disease won't spread.
This is the phenomenon of [Herd immunity](https://en.wikipedia.org/wiki/Herd_immunity).
Essentially the removed individuals act as a buffer that extinguishes the disease before it can move too far.
This is why immunisation can be so effective; if you can just get most of the community you can eradicate the disease.

# Fitting the model

Epidemiological data is notoriously unreliable.
Suppose that someone contracts the disease and is infectious.
They may not have any symptoms, and not even realise they are infectious.
Even if they do have symptoms, they may not go to a doctor for treatment and just recover on their own.
Even if they do go to a doctor, the doctor may not choose to administer a test.
Even if the doctor does choose to administer a test, it may come through as a false negative.
The only cases we can detect are the tests that came back positive.

We can only ever see the tip of the iceberg of an infectious disease, and many people will be infectious that we won't have on record.
This could be systematic; for example in areas under-served by doctors, or for a doctor who chooses not to administer a type of test.
This means that fitting a model is more than just throwing some equations in, but requires careful thinking of how the data was collected and what evidence there is to its completeness.

Because these diseases spread exponentially, missing a few cases at the start means severely underestimating the scale of an epidemic.
Unfortunately there aren't any magic bullets here; the best thing to do is contact tracing of the cases you have to try to catch as many as you can.
This can be really hard in practice though; even if individuals are willing they may not recall everyone they've been in contact with.

It's also difficult because this process can take a long time.
From being infectious to getting a positive result can take weeks.
This means any policy change to impact the basic reproduction number won't be seen for weeks; but in this time the number of people who are infected is growing rapidly.
And even then it may be hard to see through the noise and instability.

# More complex models

The SIR model I've presented here is very simple, and I doubt any epidemiologist is using it for real problems.

SIR apperently originated in [Kermack-McKendrick Theory](https://en.wikipedia.org/wiki/Kermack%E2%80%93McKendrick_theory) which modelled different recovery times and transmission rates for different age groups.
Similarly you could think of other covariates that impact transmission and estimate parameters for them separately.

There are different mechanisms by which people lose immunity (or the pathogen mutates) which is covered in the SIRS model, which has a pathway from removed back to susceptible.
You could also add other states, such as a carrier state for people who are still infectious but asymptomatic, like the famous [Typhoid Mary](https://en.wikipedia.org/wiki/Mary_Mallon).
Some of these variations are listed in the [Wikipedia page on comparmental models](https://en.wikipedia.org/wiki/Compartmental_models_in_epidemiology).

You could also be more specific in how the disease spreads and recovery happens.
This could be an agent based model on some probabilistic contact network.
Most contact networks aren't uniform and have some highly connected individuals (who can spread the disease very far), and many low connection individuals.
You could even look at specific high-risk areas like public transport.

However I don't think these will add much more insight for understanding the underlying drivers, and will add more complexity to the model and mathematics.
The SIR model is very useful for getting a grasp on why infections spread so quickly, what factors are most important in slowing the spread, and some ballpark estimates of how quickly it will spread.