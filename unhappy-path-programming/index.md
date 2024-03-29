---
categories:
- programming
date: '2020-11-06T21:56:38+11:00'
image: /images/unhappy_path.jpg
title: Unhappy Path Programming
---

When programming it's easy to think about the happy path.
The path along which you get well-formed valid data, all your requests return successfully and everything works on your target platform.
When you're in this mindset it's easy to just check it works in one case and assume everything is alright.
But the majority of real work in programming is the unhappy paths.

While you always need to be thinking about how things could go wrong, it's much more important in web programming.
I've written a lot of batch data processing, and while you have to be careful of things like malformed data and encoding issues, you control the environment so many classes of failure are manageable.
Web programming is much more uncontrolled; your client side code could be operating on a vast number of different browsers and may not work properly on some of them, any request you make can fail due to connection issues, and the data your server receives could be maliciously crafted.
It may work in one case, but then be unusable for certain users, fail intermittently and even leave your server open to compromise by an injection attack.

To build a system that handles failures and edge cases well requires a lot of discipline or some useful guard rails.
Traditional scripting languages like Javascript, PHP and Python have shockingly few such guard rails by default, and it's easy to write code riddled with bugs that will be hit once in a while that together make an unstable application.
I'm interested in how far things like type safety, forcing dealing with errors and static analysis can remove these issues in practice.