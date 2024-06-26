---
categories:
- programming
date: '2020-12-19T21:16:13+11:00'
image: /images/entity-component-system.gif
title: What Is a Better Programming Approach?
---

When you solve a problem in code you will use some programming *approach*, and the approach you choose can make a big impact on your efficiency.
I talk about *approach* rather than language because it's more than just the language.
A project will typically only use a subset of the language (especially for massive languages like C++), some set of libraries, and develop patterns in the lanugage for working with those libraries.
The problem is it's really hard to know what approach is going to work best.

I used to write a lot of bespoke reports that took data from a database and filtered, joined and aggregated them to produce a summary.
Originally there were a few generic report types that were written in PHP, because the standard reports available through a web application were written in that.
The scripts were really hard to modify because there were many function arguments that *looked* generic, but would only work for certain cases that had been implemented in the web application; and the only way to find out for sure was to dig in to the monolithic codebase.
The team moved to writing custom SQL in Python, and some data munging occurred in both SQL and Python.
Most of the Python we used standard Python data structures and [itertools](https://docs.python.org/3/library/itertools.html) to do the transformations.
Then one of our coworkers introduced us to [Pandas](https://pandas.pydata.org/); I was initially sceptical but quickly saw how effectively it solved certain problems (even if we had to keep putting `.reset_index()` everywhere).
We adopted it and it made the type of transformations we were doing really simple, our efficiency increased and removing the low level details of implementing the transforms enabled us to think about how we could deliver more value.

It's hard to know how generalisable an increase in efficiency was.
Our team generally got more things done in our Python libraries than the PHP hooks, but that was partly due to our team being more familiar with Python than PHP.
Pandas made our team much more effective than plain Python because of the types of problems we were solving.
But perhaps experienced PHP developers could have gotten a lot more done in PHP (I doubt it, but I don't know).
We had hundreds of reports, many running on a schedule, but maybe the system would have broken down at tens of thousands of reports.

After we started using Pandas I learned about R's tidyverse, but even though I think it's much better it wasn't worth our team switching from Pandas to R.
I went to another modelling team that was working in R and started reading through [R for Data Science](https://r4ds.had.co.nz/).
The bespoke reporting team used the [Pandas pipe style](/pandas-pipe) but there are some operations that are hard to pipe, like [calculating the second most common value](/topn-chaining).
Where Pandas is an organically grown mess of operations, dplyr and tidyr are well designed minimal composible APIs.
There was a tradeoff that writing functions was harder in dplyr because of the quasiquotation style you had to adopt (it looks like this has changed now and it looks much easier).
But the team was already effective enough in Pandas (and didn't know R), had extensive Python libraries for producing standard reports, and we had experience maintaining and operating the environment.
For a completely new team it would be worth considering R, but for bespoke reporting team the benefits wouldn't offset the switching costs.

When I was using Hadoop in the pre-Spark days there were lots of ways to write jobs with different tradeoffs.
The default was using Java (or anothe JVM language) to write MapReduce jobs, but it's full of boilerplate and there's a lot of code to get simple things done.
There was [Pig](https://pig.apache.org/) which allowed writing the jobs in a much simpler high-level language, and you could use custom UDFs in Java.
There was [Hive](https://hive.apache.org/) which allowed an SQL-like interface, less flexible than Pig but more similar to other tools.
Finally there was [Hadoop Streaming](https://hadoop.apache.org/docs/r1.2.1/streaming.html) which let you run any custom program on Hadoop, with some limitations.
There were some one-off jobs I wrote in Hadoop Streaming in a couple of days that would have taken the developers months to write in Java.
But I wouldn't build our whole system on this because it would be hard to maintain and ensure the integrity you could in a large Java system.
There was a place for all these tools for different kinds of workloads from more static to more dynamic.
However when Spark came along it effectively obsoleted all of these (except parts of Hive like the meta-store); it was flexible enough to do everything and you won't hear much about the other technologies any more.
The industry decided that Spark was a *better* approach for transforming data, and the PySpark interface gave it broader adoption.

The video game industry has had a large movement from building deep Object hierarchies to building entity component systems which has made it easier to build features and optimise code.
Mick West's [Evolve Your Hierarchy](http://cowboyprogramming.com/2007/01/05/evolve-your-heirachy/) explains that game developers traditionally build deep object hierarchies.
I've seen this taught in Introductory Programming classes in Java; start with an automobile class and then subclass it with a car.
Incidentally this is a terrible way to introduce programming because it's an advanced design technique you can't really appreciate until you can build simple programs, and there are many other ways to represent code.
Indeed the problem with inheritance is that it *tightly couples the components* (which is exactly the opposite of the compartmentalisation that is taught as good software engineering practice); it's essentially an easy way to copy code between components.
This means if you make a change, like your player can now control a fly-by-wire missile, you'll need to completely restructure your hierarchy and change code in lots of places (now fly-by-wire missile needs to inherit from controllable character, but it doesn't have a health bar so we'll have to create a new mutual superclass called "controllable" and move the appropriate code around).
A different approach is to treat the entities as data, an aggregation of components, and have separate systems (or engines) that act on these components.
This approach makes it easier to build new entities with custom functionality because you just need to assign the relevant components to them and let the systems act on them.
In gaming it also can make it much faster because you can allocate all the entities together and process them in parallel when running a function, rather than calling out to lots of mutable objects where concurrency is hard.
This kind of programming has it's own challenges (such as communication between entities) but has been very successful; see [game programming patterns article on components](http://gameprogrammingpatterns.com/component.html) or [games from within's article on data oriented design](http://gamesfromwithin.com/data-oriented-design) for more details.
Many game studios found it beneficial to make the cultural and code switch to an entity component system.

Because of the context dependence it's very hard to show that a programming language of feature increases productivity, as [Crista Videira Lopes' argues in an article on research in programming languages](http://tagide.com/blog/academia/research-in-programming-languages/).
She mentions that people make claims like Haskell programs have fewer bugs because of its features, without providing any evidence.
In my experience looking at [Pandoc](https://pandoc.org/), one of the most used Haskell programs (along with [ShellCheck](https://www.shellcheck.net/)), a lot of its code is complex string transformations which is a class of errors Haskell can't help with (although it may help with many others).
Even if Haskell does lead to fewer bugs, how much does it slow down development?
Maybe paying a cost of unit tests and occasional bugs is less costly than trying to frame everything in terms of pure functions and jumping through monadic hoops everywhere.
For a large development team there's the additional issue that there isn't a huge supply of Haskell developers, and they would likely have to support training people in the language which is an additional cost.
For many teams Haskell may actually decrease productivity, rather than increase.
I have heard a lot of stories of teams of Java developers moving to Scala writing "Scava code"; that is still using Java idioms rather than adoping functional programming techniques that Scala advocates.
The kind of culture change of moving to a language like Haskell may be really difficult for some developers.

There are definitely times when changing your programming approach could make your team dramatically more effective.
As far as I know there are no rules around this, largely because it depends on the team and their experience as well as the problem being solved.
Software engineering is all about tradeoffs and you have to try to find what approach is best for your situation.
In my personal experience I find object oriented approaches great for writing graphical user interfaces and functional approaches best for transforming and handling data.
I really like R for analysis, but will move to Python for transforming data, and bash for quickly doing some file manipulation and SQL for extracting data and aggregating datasets too large to work with locally.
But I'm left wondering is there a better way?
This is the best reason in my mind to learn [new programming languages](/programming-languages-2020), even esoteric ones, and try new libraries and read about different programming techniques.
One day actors or logic programming or a custom DSL or [propagators](https://groups.csail.mit.edu/mac/users/gjs/propagators/) may make a hard problem easy to solve.
But it's really hard to know what works until you have experienced (that is, have tried it a few places and gotten it wrong a couple of times).