---
categories:
- insight
date: '2020-11-16T21:20:38+11:00'
image: /images/envelope.png
title: Energy Desnsity to Launch into Space
---

This is from Sanjoy Mahajan's [The Art of Insight](https://mitpress.mit.edu/books/art-insight-science-and-engineering) Problem 1.11

> Estimate the energy in a 9-volt battery. Is it enough to launch the battery into orbit?

I have already [(mis)estimated the energy of a battery](/energy-9v-battery), but looked it up as 500 mAh.

# Energy density required to launch into space

To launch into space you have to exchange energy to counteract the change in gravitational energy (at least, you'll need more for air resistance).
The gravitational energy is $\frac{G M m}{r}$; to estimate it requires the gravitational constant, the mass of the Earth and its radius.
But we can refactor this as $r \left( \frac{G M}{r^2} \right) m$, where the first term is the radius of the Earth ($6 \times 10^6 \rm{m}$) and the second term is the acceleration due to gravity at the Earth's surface, around $10\ \rm{m}\,\rm{s}^{-2}$.
So the energy required to launch into space is $6 \times 10^7\ \rm{J}\,\rm{kg}^{-1}$ or about 60 kJ/g.

# Energy density of battery

A 9V battery has 500 mAh, and weighs about 50g.
A mAh is 3.6 J, so a 9V battery at 10 mAh/g is about 35 J/g.
It's nowhere near energy density enough to launch itself into orbit.