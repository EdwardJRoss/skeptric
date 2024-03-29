---
categories:
- data
- legacy code
date: '2020-09-16T22:07:00+10:00'
image: /images/dag.svg
title: Dataflow Chasing
---

When making changes to a new model training pipeline I find it really useful to understand the dataflow.
Analytics workflows are done as a series of transformations, taking some inputs and producing some outputs (or in the case of mutation; an input is also an output).
Seeing this dataflow helps give a big picture overview of what is happening and makes it easier to understand the impact of changes.

Generally you can view the process as a directed and (hopefully) acyclic graph.
Some data goes in, goes through some processes, and some data comes out.
While this process is easier if the transformation is done via [functions](/comment-to-function) (or even declaratively), the process is conceptual and it's possible to draw out even if it's through a big chunk of spaghetti (though if there's a lot of coordinated state changes involved you may also need a state transition diagram).

Typically the best place to start is with the outputs and trace your way back.
Looking at the outputs what input data and transformation is used to produce it?
For each of those inputs what are used to produce them?
Get a sketch of the flow at a high level and then try to drill into the pieces you're most interested in.
This is a tedious process, but you can eventually get a dataflow diagram, at least to the point where you understand the impact of the system you're changing.

While you could potentially automate this the process of drawing these diagrams is useful for getting across the codebase.
You'll start to notice patterns of how it's written, what sorts of techniques are used and an idea of the data involved.
This is something that just takes time; and tracing dataflow is a useful way to do it.

When you want to zoom in on a particular process you can always serialise (a sample of) the data or debug (for example in Python with [pdb](/pdb)). 
This is a good way to test your assumptions; think of what you expect the data to look like or some assertions you think should be true.
When the processes are hard to understand looking at these snapshots can help clarify what's going on.

This then helps give a framing of the change you want to make; what are the impacts going to be?
It's simply chasing through the dataflow diagram; what data do you need from upstream, and what changes need to be made downstream.
If you can break the change into small steps only touching one or two processes at a time it's much easier to make.
The big picture dataflow can help identify opportunities to unify repeated processes that would all need to be changed together.