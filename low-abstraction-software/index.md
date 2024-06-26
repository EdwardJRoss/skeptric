---
categories:
- programming
date: '2022-06-16T19:32:01+10:00'
image: /images/hanoi-big-picture.png
title: Low abstraction software
---

I spend most of my time with software barefly aware of the towers of abstraction below.
When I click a link in my web browser it sends a HTTP request down through to the kernel which sends the appropriate bits to a network card that are sent to my router, out through to my ISP that does more work to resolve the domain to an IP address to route the query to.
That IP is running a server that interprets the request and sends bits back over the network eventually being routed by the kernel back to the browser.
Most browsers run a separate process per tab which includes parsing the HTML and CSS to produce a layout, and interpreting the Javascript to further manipulating the DOM and sending more HTTP requests.
I'm sure my explanation here is wrong because I really don't understand how it all works.

But sometimes these [abstractions leak](https://www.joelonsoftware.com/2002/11/11/the-law-of-leaky-abstractions/) and you are forced to understand the layers.
This is especially true as a developer where you are exposed to a plethora of tools to do the simplest things; Integrated Development Environments, source control systems, build tools, package managers and containers, debuggers.
Moreover you have to interface across multiple systems; if you're running PySpark you've got JVMs interacting with Python and its libraries written in C distributed across many machines - it gets hairy.
This in particular puts up a barrier for new developers, or part-time developers; to make a simple change to some code you need to understand so many things.

Is there a better way to build these abstractions, with fewer layers and less code?
This requires tradeoffs; with speed of execution, functionality, hardware support, or integration with other environments.
But there are some systems that make these tradeoffs and show other ways of doing things.

# Building from the ground up

Building up on baremetal systems lets you understand how everything works down the CPU.
However connecting to the plethora of peripherals is difficult and so they often only work on certain hardware and have limited support.
Alex Chadwick's [Baking Pi tutorial](https://www.cl.cam.ac.uk/projects/raspberrypi/tutorials/os/) works on a Raspberry Pi v1 from baremetal and can control LEDs, drawing to a screen and taking keyboard input.
Kartik Agaram's [mu](https://github.com/akkartik/mu) is a low level computing environment for x86 machines with some apps and ability for input and output, written in a way to be understood.
More ambitious is [Project Oberon](http://www.projectoberon.com/) by Niklaus Wirth and Jürg Gutknecht which builds a whole operating system on RISC-V in a Pascal-like language including it's own form of networking.
Finally it's worth mentioning [nand2tetris](https://www.nand2tetris.org/) which simulates layers of abstraction from a NAND gate up to a programming language.

# Language with leverage

Programming languages tend to be large complex things with many corners, but there are some small well-crafted languages with high leverage.
Lisps are especially syntactically simple lanugages, and in particular [scheme](https://en.wikipedia.org/wiki/Scheme_(programming_language)) and [racket](https://racket-lang.org/) are simple languages with powerful languages with macro facilities (code that writes code) and control flow with continuations.
Emacs lisp and emacs are a particularly striking example of a malleable system that is highly adaptable (although emacs itself is quite complicated!)
[Lua](https://www.lua.org/) is a very compact language built mainly around the table data structure, a hybrid between a hashmap and a list, which is relatively flexible and has good C interoperability; Kartik Agaram built the [teliva](https://github.com/akkartik/teliva) interpreted environment on top of it.
Then there are concatenative stack languages like [Forth](https://en.wikipedia.org/wiki/Forth_(programming_language)) and [Factor](https://factorcode.org/) which can build from low level operations to high level functions with minimal syntax.
[Swift](https://www.swift.org/) build from quite low level LLVM code, for example the [builtin Math](https://github.com/apple/swift/blob/main/stdlib/public/core/BuiltinMath.swift) trigonometric and exponential functions up to a high level.

# Pragmatism

At the end of the day it's very *useful* to be able to interact with other systems.
Writing drivers for every network cards or USB is non-trivial in any system because it comes down to a lot of details, and if you can leverage Linux or BSD kernel code that's great (or maybe we should build our own hardware?).
And languages with leverage often have too much freedom, making it harder to switch to other people's code bases and integrate between systems.
But maybe in all this there are simpler ways of making software from the low to the high level that can get more people building software.