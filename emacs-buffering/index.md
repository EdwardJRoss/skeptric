---
categories:
- emacs
- wsl
date: '2020-04-16T08:05:23+10:00'
image: /images/blank_emacs.png
title: Using Emacs under WSL
---

Getting Emacs to work nicely on a Windows system can be a challenge.
You can install it natively (although getting [all the dependencies](https://emacs.stackexchange.com/questions/3874/easiest-way-to-install-emacs-windows-support-libraries) is a challenge), but many packages require libraries or utilities that are hard to install or don't exist on Windows.
The best solution I have found is using Emacs under the [Windows Subsystem for Linux](https://docs.microsoft.com/en-us/windows/wsl/about) (WSL) with [Xming](https://sourceforge.net/projects/xming/).

However if you run Emacs 26 or greater after starting Xming with `XLaunch` you're faced with a blank screen and can't see any writing on Emacs

![A blank Emacs screen](/images/blank_emacs.png)

This is because [double buffering is enabled by default](https://emacs.stackexchange.com/a/42440) and Xming (and VcXsrv) don't support this.
I added the following to my `.emacs` to work around this:

```emacs
(if (and (getenv "PATH") (string-match-p "Windows" (getenv "PATH")))
    (setq default-frame-alist
          (append default-frame-alist '((inhibit-double-buffering . t)))))
```

It tries to detect if I'm on a Windows machine by [looking for Windows on the Path](https://emacs.stackexchange.com/a/47785), and then disables double buffering.
This means it will continue to work the same when I switch to my Linux machine.

This isn't perfect; if you switch to a Windows X server that can support double buffering you could delete this line, but I don't know if that's possible to reliably detect.

## What about Terminal Emacs?

Another solution is to run Emacs in terminal mode (e.g. `emacs -nw`).
I found using [Windows Terminal](https://github.com/microsoft/terminal) that I'd see lots of artifacts when editing code like below:

![Display artifacts in Emacs](/images/emacs_terminal_artifacts.png)

This is likely because I'm using a lot of dynamic features (like relative line numbers, matching bracket highlighting, etc.).
I could always fix it by using `redraw-frame`, and maybe I should have added a hook to run that after every cursor move (I'm not sure if there would be any performance implications though).

The other issue is without a shared clipboard I would always have to copy by highlighting the text with my mouse and typing `C-S-c`, and for pasting I would use `S-Insert`.
However because it didn't know I was pasting it would reformat the text (and I don't know what the equivalent of Vim's `:set paste` is).

I probably could have configured it to work, but it's much nicer under Xming with a clipboard shared with Windows and a fully functional GUI.