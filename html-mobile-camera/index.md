---
categories:
- programming
- whatcar
date: '2020-11-04T14:27:38+11:00'
image: /images/mobile_phone_camera.jpg
title: Activating Mobile Phone Camera from HTML
---

Building a web application is great because, if it is well built, it can be accessed across many operating systems.
But sometimes you want to access particular aspects of the device; for example take a picture from a mobile camera.
It turns out this is easy to do on many systems in HTML.

A year ago I built [whatcar.xyz](http://www.whatcar.xyz) which classifies a photo of an Australian car with its make and model.
I hacked it together in a week, and it was featured on the [2019 fast.ai course](https://youtu.be/MpZxV6DVsmM?t=238), and haven't worked on it much since.
I'm now looking at making some usability and data capture improvements.

Originally when you select the car photo to classify it opens up a file selector.
On desktop this makes sense, but on a mobile device you probably want to take a photograph straight away (at least as default).
How can you open up a camera for a photo?

I found a [Stackoverflow answer](https://stackoverflow.com/a/44264339) that gives a simple solution; adding `capture="camera"` to the input element accepting images.

     <input type="file" accept="image/*" capture="camera">

The post claims it works on Android from 4.0 and iPhone OS6 and newer, and on testing it works on modern versions of both.
However it doesn't give references for this so I dug a bit deeper.

The capture attribute it covered in [MDN Documentation](https://developer.mozilla.org/en-US/docs/Web/HTML/Attributes/capture), which links to [the specification](https://w3c.github.io/html-media-capture/#the-capture-attribute).
In fact by the specification the only valid values of capture are "user" (self-view camera), "environment" (viewing out from the phone), "left" and "right".
Anything else goes to the implementation specific default; so in this case the solution works by accident.
It would be better to explicitly define the outward camera for photographing a car:

     <input type="file" accept="image/*" capture="environment">
     
The specification is really useful; it's even got an example of [how to upload a file to a server and display client side](https://w3c.github.io/html-media-capture/#example-5).
This is great because I've found upload is sometimes failing and display isn't working with my existing javascript (which is the next thing on my list to fix).

The specification is all well and good, but does it actually work on real devices?
The [W3C tests](https://w3c.github.io/test-results/html-media-capture/all.html) show it works, in particular with images and environment, on Chrome 64 for Android and Safari 11.1.2 for iOS.
According to [Caniuse it's supported on 97% of (tracked) mobile devices](https://caniuse.com/?search=htmlmediacapture), although it says it doesn't work on Firefox for Android when I have manually verified it can.

On the desktop and other browsers that don't support it the input falls back to a file picker, which is exactly what I would want.
Just by adding the `accept` and `capture` attributes my website now works how I would want.