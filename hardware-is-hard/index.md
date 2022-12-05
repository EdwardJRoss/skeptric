---
categories:
- programming
date: '2021-05-14T22:03:21+10:00'
image: /images/raspberry_pi.png
title: Hardware Is Hard
---

I've been revisiting baremetal Raspberry Pi programming (with [Alex Chadwick's Baking Pi tutorial](https://www.cl.cam.ac.uk/projects/raspberrypi/tutorials/os/), although there are [plenty](https://github.com/dwelch67/raspberrypi) [of](https://jsandler18.github.io/) [others](https://www.valvers.com/open-software/raspberry-pi/bare-metal-programming-in-c/)).
It really highlights how much I take for granted.
I spend a lot of time in languages like Python and R processing data, without much understanding of the interpreters written in C, let alone thinking about the compilers that turn those interpreters and libraries into assembly, or how that assembly executes.

Getting an LED blinking feels like a huge accomplishment.
Reading the [BCM2835 Arm Peripheral Documentation](https://www.raspberrypi.org/documentation/hardware/raspberrypi/bcm2835/BCM2835-ARM-Peripherals.pdf) gives most of the information you need (with a little black magic to get started), but it's rather dense.
There's a complete lack of feedback when something goes wrong, and the cycle between write, compile, save to SD card, insert in raspberry pi and book takes a little time.
The whole process takes a lot of persistence.

Then showing something on the screen requires understanding the interface between the GPU and the CPU, but the closest thing I can find to an official resource is a [Wiki pages in Raspberry Pi Firmware git repository](https://github.com/raspberrypi/firmware/wiki/Mailboxes).
I suspect a lot of experimentation and reverse engineering have gone into making this work.

Finally taking input from a USB keyboard requires interfacing with the USB which doesn't have much documentation.
Moreover the USB specification is huge and complex; it's got a lot of flexibility but the cost is software complexity.
I suspect a lot of this came from looking at the open source Linux implementation as a starting point.

It really makes me consider the amount of time, engineering and coordination that has gone into something like Linux (or the BSDs).
Implementing the hardware implementations for different types of computers and systems, all the drivers for external peripherals (so when you plug in a printer you can print), interfacing with graphics cards, as well as managing processes and everything else.