---
categories:
- wsl
date: '2020-07-20T23:42:44+10:00'
image: /images/win_ubuntu.png
title: Getting Started with WSL2
---

I've finally started trying out Windows System for Linux version 2.
When [comparing with WSL1](https://docs.microsoft.com/en-us/windows/wsl/compare-versions) it's much faster because it works on a Virtual Machine rather than translating syscalls, but is slower when working on Windows filesystems.
The speed up is significant when launching processes and dealing with small files, and git and Python virtualenvs are an order of magnitude faster.
I'm still working through some of the issues of transferring.

It took me a while to try WSL2 because, as per the [introductory blog post](https://devblogs.microsoft.com/commandline/wsl2-will-be-generally-available-in-windows-10-version-2004/) I was waiting for Windows version 2004 to appear when running "Check for Updates".
But it has been 2 months since the May release and it still hadn't appeared.
So after checking there were no [known issues with my hardware](https://docs.microsoft.com/en-us/windows/release-information/status-windows-10-2004), I [manually downloaded the update](https://www.microsoft.com/en-au/software-download/windows10).

After some time the upgrade installed without any issues.
When I booted Docker for Windows prompted me to install WSL2, because it can now run through that.
I followed the WSL2 [installation instructions](https://docs.microsoft.com/en-us/windows/wsl/install-win10#update-to-wsl-2), and installed Ubuntu 20.04 LTS from the Microsoft Store as my Ubuntu version (I had Ubuntu 18.04 in WSL1).
I then set it up with WSL2 using `wsl --set-version Ubuntu-20.04 2`.

I was getting confused because `wsl --set-default-version 2` wasn't changing the default version when I checked with `wsl -l -v`.
This is because I was using a different distribution, and had to set the default distribution with `wsl -d Ubuntu-20.04`.

Finally because I'm using the excellent [Windows Terminal](https://docs.microsoft.com/en-us/windows/terminal/) I [set the default profile](https://superuser.com/questions/1456511/is-there-a-way-to-change-the-default-shell-in-windows-terminal) to WSL2.

Unfortunately there are a couple of things that don't work out of the box.
I can't get Emacs working with VcXsrv, even after following these [instructions for working with Emacs and WSL](https://github.com/hubisan/emacs-wsl); it just defaults to `-nw`.
Another issue is that I can't just run things on localhost in WSL and expect it to work in Windows, and have to deal with some [virtual networking](https://docs.microsoft.com/en-us/windows/wsl/compare-versions#accessing-network-applications).

But after a few years on WSL it's amazing to have much faster speeds and have things like emacs magit and Python virtualenvs feel usable again.