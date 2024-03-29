---
categories:
- programming
date: '2020-08-30T08:00:00+10:00'
image: /images/learning_to_code.jpg
title: Teaching Programming by Editing Code
---

I've had a few discussions with people, especially analysts, about how to learn programming.
Generally I encourage them to find a project they want to accomplish and try to learn programming on the way.
However I really struggle to find resources to recommend because they tend to spend a lot of time teaching programming concepts from scratch.
I wonder if a better way to teach these things would be to start with code that's close to what they want to accomplish, and get them to edit it.

When someone asks me for advice on what resources to recommend, I carefully consider their background and where they want to get to.
Often analysts will have come across things like `if` statements through Excel or Tableau, and have an understanding of what they want to do to the data, and how to break down problems.
This actually means they are quite capable of picking up many programming concepts, even if they've never studied programming.

But more important than where they are is where they want to get to.
Generally people want to learn programming to do something, and I've seen a variety of examples:

* summarising themes from free text survey responses
* performing custom statistical tests on skewed data, that their testing tools can't handle
* fitting more models to better inform the outcomes of a decision
* automating regular reports that are currently painfully extracted from multiple sources

Depending on what they want to do, the actual materials that are relevant will vary.
Some people want to transition into a software engineering career, and that will often be a different path again.

However I get frustrated that the vast majority of learning to program materials spend a lot of time on dry theory.
There are typically chapters on every different type of programming construct (types, string, conditionals, loops, vectors, objects, ...), with little motivation as to *why* they study them.

There's a learning concept called the [Whole Game](https://www.gse.harvard.edu/news/uk/09/01/education-bat-seven-principles-educators) which focuses on teaching by getting people to actually play the game rather than just learn the rules.
By getting students to do a small version of what they're trying to accomplish, and getting them to learn themselves, is extremely motivating.
I've seen this myself in the [fastai](https://www.fast.ai) course where I learned this concept.

For example take the problem of summarising themes from free text survey responses.
You could start by present a basic [topic model](/topic-model-bootstrap) to extract themes from a text.
Then get them to point this topic model at their own text; this alone can often be quite a challenge to extract the text and put it in the right format.
Then you could give them some options on how to customise it; to see how changes make the model better or worse.
This can start as simply as learning to edit the code to change constant values.
Then maybe they need to find ways to print out words and sentences of different topics by changing an index.
A for loop would let them display all the topics at once, or maybe process multiple surveys.
Maybe converting the text to lowercase before fitting the model would lead to better results?

This is a basic example, but I really think if someone is really motivated to build a good theme summary automatically they could learn a lot from doing this.
And it would enable them to better learn how to solve related problems.
When they start hitting some limits they might start wondering how the model actually works, and that could lead to much deeper learning.

Depending on the level experience you may want to hide some of the complex logic (one of the wondrous things about programming is that you can do this through functions).
How you hide the complexity, and how much you hide, depends on where you want the student to focus.
This means thinking a bit about where they are, and how they can learn in small steps.
Also it's really hard to do this - I find sitting with someone as they work through the problem can be really helpful because they'll hit issues you didn't even think of.
Some gentle encouragement, hints, or occasionally fixing weird problems can be very helpful in getting them through the process.
But as much as possible you want them to drive their own learning.

I haven't seen this idea of teaching programming by editing code to be widely taught.
But in practice it's how most programmers actually work.
I'm much more likely to be maintaining and extending some existing code than I am to be writing my own.
Even when I'm writing my own, it's not unusual for me to look at tutorials or Stack Overflow for how to solve particular problems.
Similarly I think this would be a valuable way to learn how to code.

That's not to say the fundamentals aren't important.
But you're much better off learning why they're useful first, before learning all the intricate details of what they are.
I've learned a lot by taking two courses in algorithms and data structures and a course in compilers, by reading (most of) the [Structure and Interpretation of Computer Programs](https://mitpress.mit.edu/sites/default/files/sicp/index.html) (one day I'd like to look at [How to Design Programs](https://htdp.org/)).
But I started learning programming by trying to build games, first in various frameworks and then in Java.
I then learned a bunch more trying to do Physics simulations and calculations.

Really the biggest risk isn't that a student never reads the specification; it's that they lose the motivation and give up.
As Peter Norvig points out, learning programming [takes years](https://norvig.com/21-days.html), but it starts with small steps.
