---
categories:
- programming
date: '2021-05-25T20:53:02+10:00'
image: /images/arm_draw_character.png
title: Code Optimisation as a Trade Off
---

I've been writing some [ARM Assembly](https://github.com/EdwardJRoss/baking-pi) as part of a [Raspberry Pi Operating System Tutorial](https://www.cl.cam.ac.uk/projects/raspberrypi/tutorials/os/), and writing in Assembly really forces me to think about performance in terms of registers and instructions.
When I'm writing Python trying to write concise code leads to breaking a problem into small functions or methods (and using idioms like list comprehensions).
The same drive for concise code in Assembly leads me to reduce the number of instructions used and the number of registers, but even though it *feels* like its making things more efficient it may have negligible actual impact.
However these micro-optimisations often lead to real engineering trade-offs with concepts like safety, modularity or modifiability.

In [writing character bitmaps](https://www.cl.cam.ac.uk/projects/raspberrypi/tutorials/os/screen03.html) to a framebuffer we need to scan through each row and column of the bitmap.
In the tutorial Alex Chadwick suggests instead of keeping track of the row, by aligning the font bitmaps appropriately, we can just track the last bits of the address.
This works well and removed a variable and we can use one less register.
However if we want to use a bitmap with more rows we may need to change not only the code but also the font alignment; it's been made more difficult to maintain.

The optimisations for writing character bitmaps *look* like they create more efficient code, but I haven't profiled them.
What I do see is they end up calling the subroutine `DrawPixel` once for each bit with value 1, and so the efficiency really depends on this routine.
And [looking into it](https://www.cl.cam.ac.uk/projects/raspberrypi/tutorials/os/screen02.html#dots) we can see that at each step it does a bunch of loading data from memory and comparisons that are the same each time it's invoked.
These could potentially be much more impactful on performance than the register we saved above.

Unfortunately there are tradeoffs trying to reduce the amount of work `DrawPixel` does.
One task it has is to check whether the coordinates of the pixel lie in the width and height of the framebuffer.
We could remove these to save time in an `UnsafeDrawPixel`, and move the range checking logic into `DrawCharacter` where it could be done once.
But having two implementations, `DrawPixel` and `UnsafeDrawPixel`, duplicates the logic making it harder to change.
We could combine the implementations by separating out the checking logic into a subroutine, but it leaves us with more risk of writing outside the framebuffer.

Another opportunity would be to completely inline `DrawPixel` so we can keep the FrameBuffer pointer in a register rather than reloading it at each pixel.
But this trades off modularity for speed; we have to duplicate the pixel drawing logic which increases the code complexity and gives more places to get things wrong.
There are potential other opportunities for optimisation, but they require breaking other things such as the ABI making it harder to inter-operate with other code.

Ultimately there are often engineering tradeoffs for optimisation; we give up other desirable properties of code like generalisability, modularity or safety.
And almost always we want to focus on those - writing a safe and easy to maintain system is hard enough in the good cases.
However where we really need speed, in the slowest part of the program (which can only be reliably identified with profiling) it can be really worth investing in optimisation, even if it means giving up on some of these other things.

As Knuth said in [Structured Programming with Goto Statements](http://web.archive.org/web/20130731202547/http://pplab.snu.ac.kr/courses/adv_pl05/papers/p261-knuth.pdf):

> Programmers waste enormous amounts of time thinking about, or worrying about, the speed of noncritical parts of their programs, and these attempts at efficiency actually have a strong negative impact when debugging and maintenance are considered. We should forget about small efficiencies, say about 97% of the time: premature optimization is the root of all evil.
> Yet we should not ot pass up our opportunities in that critical 3%.
