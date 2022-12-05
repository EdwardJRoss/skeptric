---
categories:
- data
- physics
date: '2021-03-03T21:30:13+11:00'
image: /images/downey_modeling_framework.png
title: The Way of the Physicist
---

A large number of the physicists I trained with are now data scientists, and it's not uncommon to meet a data scientist who trained in Physics.
Part of this is because there's not a lot of physics jobs, especially in Australia.
But another reason is that the training we get as physicists is very similar to what you need for data science.

David Bailey, a physicist from the University of Toronto, has [objectives for their undergraduate Physics program](https://www8.physics.utoronto.ca/~dbailey/UGPhysicsGoals.html) which describes "the Way of the Physicist":


> *  construct mathematical models of physical systems
> *  solve the models analytically or computationally
> *  make physical measurements of the systems
> *  compare the measurements with the expectations
> *  communicate the results, both verbally and in writing
> *  improve and iterate

I never use the topics I trained for in Physics day to day, but I use the conceptual framework all the time.

## Construct mathematical models of physical systems

Constructing a conceptural model is a crucial step in any data science problem.
Any [analysis needs to change a decision](/analysis-decision) which means it needs to connect input decisions to outcomes.
This connection is a model, which I'll call a *conceptual model* to distinguish from an *algorithmic model* that is directly fit to data.

Even when using a blackbox algorithmic model it's important to have a conceptual model of the underlying process.
The process may be broken down into several algorithmic models.

A good conceptual model will identify the key drivers of the outcomes, outlining the key relationships.
It's always useful to consult with domain experts who understand that process well, because they often have a good intuition of what is important.

## Solve the models analytically or computationally

When you have a model you will want to be able to solve it.
Physics textbooks primarily focus on problems that can be solved from first principles, which can be rarely done in data science.
However it is possible to come up with ballpark estimates, and solve "easy cases".

A large part of the trick to solving models is coming up with models that are solvable.
Models should be as *parsimonious* as possible, focusing on the most significant effects first.
If you put in too many effects you lose the insight, and can't see the forest from the trees.

Physics has a really good selection of models that are simple, but insightful.
In Statistical Mechanics an [ideal gas](https://en.wikipedia.org/wiki/Ideal_gas) where the particles don't interact at all is a useful model for many real gasses.
In Mechanics the [driven harmonic oscillator](https://en.wikipedia.org/wiki/Resonance#The_driven,_damped_harmonic_oscillator) is useful for understanding resonance, which can [collapse bridges](https://www.youtube.com/watch?v=XggxeuFDaDU).
In Quantum Mechanics the structure of atomic shells can be understood by ignoring interactions between electrons, which is a useful (but very wrong) model for understanding the chemistry of atoms.

Analytic models are powerful because you can explore them in detail, but they have their limits.
For many sophisticated models we need to perform simulation, optimisation, and model fitting.
Model fitting is what many many people think of in data science; extracting relations between an outcome and input variables.

## Make Physical Measurements of the Systems

The crucial connection between theory and reality is in the act of measurement.
In data science you can often collect data off the "data exhaust" of existing processes, but this is really messy.
Understanding how to collect data, whether it's digital tracking, surveys, or sensor arrays is crucial to connect the model with reality.

Even if you don't directly make the measurements it's very important to understand *how* the measurements are made.
Measurements are always imperfect, and you're rarely measuring the quantity you actually want to measure.
Understanding the inferences, approximations, and errors that occur in the measurements help understand how to assess your model.

## Compare the Measurements with the Expectations

In data science this can occur offline by checking a model against a test set (or with cross validation), to see the generalisation error.
But the best tests are always in an online setting; performing the action and seeing whether it meets expectations.
When comparing options the cleanest way is an A/B experiment.

An important premise of this is you should always *have* an expectation from your prior modelling.
You shouldn't be deploying models, running experiments, or making important data-driven decisions without having an expectation of what the outcome should be.

## Communicate the Results, Both Verbally and in Writing

A data scientist, like a physicist, is part of a community.

You have to be able to communicate your findings with stakeholders to influence decisions and get better outcomes.
The communication should be tailored to your audience.

## Improve and Iterate

Whenever there's a difference between the measurements and expectations there's an opportunity to improve.
Whether it's improving the models, the measurements or even the communication, to get the desired impact.
Starting with a simple model and iterating on it is often a much better path than starting with an overly complex model, because you get faster feedback on what is important.