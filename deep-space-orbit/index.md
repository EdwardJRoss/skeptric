---
categories:
- insight
date: '2020-11-17T21:41:58+11:00'
image: /images/envelope.png
title: Energy to Orbit vs Launch into Deep Space
---

This is from Sanjoy Mahajan's [The Art of Insight](https://mitpress.mit.edu/books/art-insight-science-and-engineering) Problem 1.11

> Estimate the energy in a 9-volt battery. Is it enough to launch the battery into orbit?

I tried to answer this [with the energy density required to launch into deep space](/energy-density-launch).
But this is different to going into orbit; how much energy is required to get into low Earth orbit?

# Low Earth Orbit

A low orbit has to be above the height of the atmosphere (otherwise will require propulsion to overcome atmospheric friction), and so is typically above 300 km.
Given the radius of the Earth is around 6000 km, the orbit is only 5% further from the centre than the ground.

In orbit the gravitational acceleration, $\frac{GM}{r^2}$ needs to balance the centripetal force to curve the objects path, $\frac{v^2}{r}$ (which is the only function of orbital speed and distance that has the right units, up to a constant).
The gravitational acceleration in low Earth orbit is about 10% lower than at the surface, so still around 10 m/s².

So the kinetic energy is $\frac{1}{2} m v^{2}$, and so the density is $\frac{1}{2} r \frac{GM}{r^2}$, that is about $3 \times 10^{7} \ \rm{J}/\rm{kg}$.

To get into orbit the change in gravitational energy is $\frac{GM}{r} - \frac{GM}{1.05 r} \approx 0.05 \frac{GM}{r}$.
That is it's about 5% of the kinetic energy, and so is a small adjustment.

So the energy density required to get into a low Earth orbit is around 30 kJ/g, which is about half the estimate we had for launching into deep space.
So even for a low Earth orbit the energy density required is about 1,000 times more than in a 9V battery (at 35 J/g).