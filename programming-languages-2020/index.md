---
categories:
- programming
date: '2020-09-21T08:00:00+10:00'
image: /images/programming_languages.png
title: Programming Languages to Learn in 2020
---

> A language that doesn't affect the way you think about programming, is not worth knowing.
>
> [Alan Perlis](http://pu.inf.uni-tuebingen.de/users/klaeren/epigrams.html)

I spend a lot of time programming in Python and SQL, some time in Bash and R (or at least tidyverse), and a little in Java and Javascript/HTML/CSS.
This set of tools is actually pretty versatile about getting things done, but is fairly narrow from a programming concept perspective.
Once in a while I think it's useful to broaden the programming frame to understand different ways of doing things; even if you still stick to the same few languages.

I've spent significant time in the past programming in Lua, Scheme, and Clojure and spent some time programming in [J](https://www.jsoftware.com/), Octave/Matlab and ARM Assembly.
Each of these languages has contributed to how I think about programming and helps me design programs today.
I wanted to reflect a bit on what languages I'd like to learn in the future.

With this in mind my primary goal is *learning new ways of thinking*, and secondarily ones that have a good community and ecosystem that can solve real problems effectively.
I'm not thinking too much about employability, as that changes with language fashion every few years.

Based on some reading I've picked a few categories ang languages I would like to learn, and why I picked it over alternatives.

* Typed Functional: F#
* Concurrent: Elixir
* Efficient: C
* Search: ???

# Typed Functional Language: F#

Typed functional languages claim to be very robust.
It's really easy to make code that works most of the time but then fails in obscure ways on edge cases (as I've learned well in Bash and PHP).
Typed functional languages prevent a lot of categories of error through composing simple pure functions and statically verifying that the functions fit together and every alternative is handled (and not forgotten).

They have much richer types, and ways of creating types, than a language like Java that remove a lot of the boilerplate and allow more expressive code.
Immutability reduces the amount of state complexity and makes them easier to test (without extensive mocking).
Because of the static type checking they claim to be much easier to safely refactor than languages like Clojure.

I'm picking F# because it looks interesting and quite different from what I know.
It's learned a lot of lessons from OCaml and built on top of it.
It seems quite elegant, and has access to lots of useful libraries from .NET (though I have heard rapid changes to .NET can make for painful transitions).

## Resources

There are a [lot of F# resources](https://fsharp.org/learn/), but I'm particularly interested in [Domain Modeling Made Functional](https://fsharpforfunandprofit.com/books/) from [F# for Fun and Profit](https://fsharpforfunandprofit.com).
This sounds like a great resource for a way of thinking of how to design programs and implementing it in F#.

## Alternatives

* OCaml, which is similar to F#, but since F# came later (and learned some lessons) and has access to .NET libraries it seems to me a better choice to start.
* Haskell, which I have some passing experience with, but I've feel like you end up jumping through monadic hoops for the sake of purity (maybe it's worthwhile; I'm yet to be convinced).
* Scala but it seems a bit more multiparadigm and to learn new concepts F# seems like a better choice.

These are all interesting languages I'd like to learn more about; but I'd start with F#.

# Concurrent Language: Elixir

I spend a lot of time programming in Python, but the GIL and library support makes concurrency programming painful.
But being able to have lots of lightweight concurrent operations is really useful for some kinds of applications, especially APIs.

I'm really interested in Elixir because it's built on BEAM, the library for Erlang.
Erlang has a reputation for building robust distributed systems (in fact built by Ericsson for telecommunications) by having lightweight actors that robustly define how they handle communication errors.
This sounds like a totally different way of thinking very useful for programming distributed systems.

## Resources

From the [Elixir Forum](https://elixirforum.com/t/which-book-to-read/16485/4) it sounds like [Elixir in Action](https://www.manning.com/books/elixir-in-action-second-edition) and [Programming Elixir](https://pragprog.com/titles/elixir16/programming-elixir-1-6/) are good resources.
I'll have to research more deeply here.

## Alternatives

* Erlang is the language that Elixir is built on, but it abstracts a lot of the syntactic sugar away. [Learn You Some Erlang](https://learnyousomeerlang.com/) is meant to be a great resource for both.
* Go allows a lot of concurrency through Goroutines, and has tiny cross-platform binaries, but I don't think I'd learn as much about resilience.
* Node.JS allows a lot of asynchronous work through callbacks, but I'm not really convinced that this style is easy to program in.

# Efficient: C

Sometimes you want a low level language for performance, or to access low level systems.
Building a very basic Operating System on a Raspberry Pi through the [Baking Pi Tutorial](https://www.cl.cam.ac.uk/projects/raspberrypi/tutorials/os/) made me think much more about how things really go on at a system level.

C is *the* systems programming language.
A *lot* of useful code is written in C, like the Linux Kernel, and knowing a bit about C lets you customise this software and write fast things yourself.
It also shares a lot of culture with C++, which is another widely used language, but is a lot harder to start with.

## Resources

I've read a lot that K&R C is the place to start.
It may not cover modern C, but gets across a lot of the initial philosophy and structure.

## Alternatives

* C++ is what most low-level numeric code is implemented in and is a very useful thing to understand. It's also got a reputation for being huge and complex and taking a long time to learn. I would go here after learning C.
* Rust sounds like it fixes a lot of the hard problems especially in C++ that lead to security vulnerabilities, while remaining low level and fast. But it's not as widely used yet, and to me it makes more sense to start with C++ before learning C.
* Julia claims to be very fast for numeric calculations, and takes a different approach to Python and R. The ecosystem is building (especially through interfacing with R and Python) and is one to watch. I've heard that by design it has great language interoperability.
* Swift is really interesting in building atomically on LLVM. As a fan of Scheme I really like how it builds from LLVM primitives (e.g. see implementations of [bool](https://github.com/apple/swift/blob/master/stdlib/public/core/Bool.swift#L245), and it's useful for building iOS applications. One worth learning.

These all sound really interesting to me - if only there was more time.

# Search Languages: ???

I really like the idea of the computer doing the hard work of solving a problem rather than me telling it how to solve it.
Search approaches are useful for this, but I'm not really sure what I'd pick here.
Some interesting logic programming languages are:

* Prolog: The most famous logic programming language. Can implement many interesting algorithms
* [Mercury](https://mercurylang.org/): Functional logic language; a little immature
* [Minikanren](https://en.wikipedia.org/wiki/MiniKanren): A way of embedding logic programming in languages
* Datalog: A logic query language, a sort of alternative to SQL

But there are other types of search techniques:

* Constraint Programming in [MiniZinc](https://www.minizinc.org/): Describing how to solve problems as optimisation under constraints. I'm also interested in the [Discrete Optimisation](https://www.coursera.org/learn/discrete-optimization) covering some of these methods.
* [Z3 Theorem Prover](https://github.com/Z3Prover/z3): An SMT Solver
* [Simulated Annealing](https://en.wikipedia.org/wiki/Simulated_annealing): An effective optimisation method for certain problems
* Proof Assistants: Like Coq, Agda and Idris

These are all things I'm interested in, but don't know enough about.
But I feel like one day I'll hit a problem (or I already have) that would be easy if only I knew them better.