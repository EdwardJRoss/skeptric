---
categories:
- linux
date: '2021-11-29T15:20:36+11:00'
image: /images/autorandr_instructions.png
title: Automatically changing display settings with Autorandr
---

I've been using Linux on laptops for over a decade and have got used to some of the rough edges.
When I boot my current laptop connected to an external monitor xrandr will extend the displays between the external monitor and the laptop screen.
I have my laptop half-shut low on my desk and so the screen is unusable, and so I open up lxrandr, which always opens up on the laptop screen, and so I awkwardly crane my neck and wiggle the mouse until I click the setting to turn off the laptop screen and press apply, then rush to press the confirmation button.

Then some time later I'll disconnect the laptop from the monitor, realise there's nothing on the screen, and plug it back in, run lxrandr again, and set the display to only the laptop monitor.

I know there are better ways of handling this.
I could spend ten minutes working out how to get my desired configurations with xrandr from the command line, and then I can blindly type into a fresh terminal without having to worry about the monitors.
Even better I could wrap it in a script (maybe called something memerable like `monitoron` and `monitoroff`).
But it happens infrequently enough that I haven't bothered.

But today it bothered me and I wondered if there was an easier way to do it.
In Windows it normally does the right thing and is quick to change configurations by pressing Windows Key and p together.
A quick search brought me to the fantastic [autorandr](https://github.com/phillipberndt/autorandr) which quickly solved the problem for me.

It automatically responds to configuration changes (like plugging or disconnecting a monitor).
It's fantastically simple to use; I could get it running in about two minutes by following the documentation.
Set up the display I want when connected to an external monitor and type `autorandr --save docked` (or replace docked with whatever you want to call it); then set up the screen and disconnect the monitor and run `autorandr --save mobile`.
Then I'm done; it automatically switches configurations whenever I connect or disconnect the external monitor in less than the time it takes to read `man xrandr`.
If you want to get your tools adopted make it this easy to get started; a quick reward is a great way to get someone hooked.

This is the awful and wonderful thing about Linux; the defaults can be clunky but it's so configurable that a few scripts can almost always solve your problem, you just need to write or find those scripts.
Autorandr seems really well designed in a UNIX style; each profile is in a folder consisting of a `setup` file (that defines the conditions when a profile would switch) and a `config` file (that defines the xrandr configuration to use).
I love that it's so observable, and its extendable to add scripts that fire before or after a configuration is triggered (which could be handy if you needed to fix a menubar of something); but you don't need to know all that to get started.

So if you ever find yourself fiddling with `arandr` or `lxrandr` to make the same configuration changes, automate it with `autorandr`.