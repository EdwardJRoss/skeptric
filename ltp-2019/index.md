---
date: 2019-10-20 22:11:38+11:00
image: /images/ltp_2019.jpg
title: Leading the Product 2019
---

I attended the excellent 2019 [Leading the Product](https://www.leadingtheproduct.com/) conference in Melbourne with aronud 500 other Product Managers and Enthusiasts.
The conference had a broad range of great talks, a stimulating networking event where we connected by sharing our favourite books on product management, and overall a energetic atmosphere.
I got something out of every talk, but here are the highlights from a data perspective.

# Find quick ways of testing difficult and uncertain hypotheses

[John Zeratsky](https://johnzeratsky.com/) talked about the [design sprint](https://www.thesprintbook.com/) for implementing a design solution in a week; from storyboarding an experience, to brainstorming solutions to prototyping and testing.
The main point is when the value of an idea is uncertain, and implementation is difficult, it's worth investing in building a "throwaway" prototype to test whether it actually works in practice.
This reminds me of [Teresa Torres approach to hypothesis testing](https://www.producttalk.org/2015/01/run-experiments-before-you-write-code/) and of the idea in [Douglas Hubbard's How to Measure Anything](https://www.amazon.com/gp/product/0470539399) of getting even a crude measure of a very uncertain value helps make much better decisions.

Another idea on these lines came from [Tom Crouch](https://au.linkedin.com/in/tom-crouch-a053b1a2) from Qantas Hotels who gave a lightning talk on using Customer Service teams to "fill the gaps" in technology by manually performing a process rather than writing code to automate it.

# Product decisions need to drive AI Products

[Sally Foote](https://au.linkedin.com/in/sallyfoote) the Chief Innovation Officer at Photobox talked about problems with their first attempt at automatically laying out customers photo books.
People use Photobox to create photo books to send to loved ones, and the process typically takes around 40 hours of elapsed time.
Sally talked about how they implemented an algorithm to automatically choose photos and layout the book, tests showed customers could create books in a much shorter timeframe.
But usage was terrible; most people opted out of the automated experience, and those that stayed in wouldn't finish their photo book.
When doing user research they found out that people actually really enjoy selecting photos, especially the cover photo, and didn't value a photo that was "picked" for them.
They changed their process to make the customer feel more involved with the creation process, while still using the same algorithms to automate the boring bits (selecting the best photo from near duplicates, sizing photos appropriately, laying them out in the right order in the book).

When they couldn't find the created date of photos the algorithm would dump them at the back of the photo book (in otherwise chronological order)
During user research they found this disjointed arrangement shocked people and made them lose trust; they then went back and checked the rest of the book to see if it made any other mistakes.

When building a data driven product you still (unsurprisingly) have to focus on the whole experience.
Automating away the things that make people engage with and value your product is a bad idea, and sometimes giving a bad prediction can make them lose all trust.
Generally it seems giving appropriate mechanisms for control is important for things considered of value (e.g. providing a list of next best actions rather than just taking an action).
They probably could have discovered a lot of this with "mock" automated layouts, rather than after launching the product.

# User Research tells a richer story than metrics alone

Sally Foote's talk covered this pretty well, but it also came from [Sherif Mansour](https://medium.com/@sherifmansour) from Atlassian.
He talked about how they drove up pages created per user by hiding "advanced" types of pages in Jira, but then found later that people weren't creating advanced types of pages and new users were compaining about the lack of functionality.
A lot of this comes down to choosing the right metrics that align to business value, rather than poor proxy metrics; but that's easier said than done and as Sherif pointed out the "right" metrics can be very hard to move (especially if you can only measure them indirectly or with very low precision).
It's always good to do user testing; the high bandwidth low volume analysis is a great complement to the low bandwidth, high volume analysis from digital analytics (which apparently they hadn't instrumented anyway...).

# Leading the Product - worth it

Overall I'd definitely go again, it was an energising conference that gave me a range of different viewpoints on what it takes to manage great digital products.

For more details of the talks checkout their [blog](https://www.leadingtheproduct.com/blog/).