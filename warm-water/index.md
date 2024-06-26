---
categories:
- insight
date: '2020-09-28T20:28:14+10:00'
image: /images/envelope.png
title: Mixing Warm Water
---

I used to have a fancy kettle that came with settings for heating water to different temperatures between 80° C and 100° C.
However it's really easy to get water at any temperature using an ordinary kettle by mixing with refrigerated water.

When you mix together two volumes of water at different temperatures their volumes add and the resulting temperature is a volume weighted average of the temperatures.
For example if you take 25mL of water at 10° C and 75mL of water at 40° C you will get 100mL of water at 32.5° C.

A fridge is typically around 3° C, and so this is the temperature of well refrigerated water.
Water boils at 100° C so this is close to the temperature you get out of a kettle.
So if you mix x mL of refrigerated water with y mL of boiled water you will get (x+y) mL of water at $\left(\frac{3x + 100y}{x+y}\right)$° C.

We can reparameterise this in terms of the ratio and total water.
The total volume is V = (x + y) and the fraction of hot water is $h = \frac{y}{x+y}$ and so the final temperature is  $\left(3(1-h) + 100h \right)° \rm{C} = (3 + 97h)° \rm{C}$

If we round down the temperature of a fridge to 0° C then this becomes a trivial equation to solve; to get water at t° C, between 0 and 100, then we require h = t.
So to get water at 80° C we would want 80% boiling water and 20% refrigerated water.
To get 50mL tepid water at 40° C we would want 40% boiling water and 60% refrigerated water, that is 20mL of boiling water to 30mL of refrigerated water.

Note that water left out will equilibrium over time to room temperature, so you should overshoot slightly for above room temperature and undershoot slightly for below room temperature.