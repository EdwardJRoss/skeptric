---
categories:
- insight
date: '2020-11-15T22:55:24+11:00'
image: /images/envelope.png
title: How Much Energy is there in a 9V Battery
---

This is from Sanjoy Mahajan's [The Art of Insight](https://mitpress.mit.edu/books/art-insight-science-and-engineering) Problem 1.11

> Estimate the energy in a 9-volt battery. Is it enough to launch the battery into orbit?

We're just going to estimate the first part.

# Battery Energy

A volt is energy per unit charge $V = \frac{E}{q}$.
To get towards an energy we need an amount of charge; the current in Ampere is the charge per unit time $I = frac{q}{t}$.
So the product $V I = \frac{E}{t}$ is energy per unit time, or power.

My smoke detector needs a 9V battery, and should be replaced every year.
If I can estimate the current the smoke detector draws I can estimate the energy.

The current drawn can be guessed with gut estimates based on my experience with current.
1 A is a lot of current, it's probably less than that, but it could be 100 mA.
1 mA is not much current it's probably more than that, but it could be 10 mA.
Guessing the geometric mean of 1 A and 1 mA gives 30 mA, or 0.03 A.

Then the Energy in a 9V battery is given by

$$E = V I t = 9 \rm{V} \times 0.03 \rm{A} \times 3 \times 10^{7} s = 10^{7} \rm{J}$$.

A common unit for energy in batteries is Watt Hours, which is 1/3600 J.
So the energy is roughly 2500 Wh, or 2.5 kWh.

# Checking

Wikipedia lists the [typical capacity of 9V batteries](https://en.wikipedia.org/wiki/Nine-volt_battery#Technical_specifications) as around 500 mAh.
This then corresponds to around 5 Wh of energy.
I've overestimated by a factor of 500.

It's easy to trace back where I went wrong; the most uncertain estimate was the current drawn by a smoke detector.
According to [energyrating.gov.au profile on Smoke Alarms](https://www.energyrating.gov.au/sites/default/files/documents/sb200405-smokealarms_0.pdf) the power drawn by a smoke alarm is less than 0.1 mW.
So dividing the power by the 9 V gives a current drawn of around 0.01 mA.

It turns out my gut was completely wrong; I didn't really know enough about currents to make a gut estimate.
I would have to think about another way to estimate the energy that doesn't rely on knowledge about currents that I don't have.