---
categories:
- wsl
date: '2020-09-04T21:03:58+10:00'
image: /images/firewall_local_ports.png
title: Fixing suddenly unable to connect to X server in WSL2
---

Today when I tried to connect to VcXsrv after running it with XLaunch it didn't work.
I'd had it working for months and so was surprised it suddenly stopped working.
The reason was simple; the IP subnet WSL2 had changed and so it was now being blocked by a firewall.

Annoyingly there is very little feedback as to *why* it can't connect to an XServer.
I went back through my previous instructions of [setting up an X server in WSL2](/wsl2-xserver), but noticed something.
When I ran `ip addr | grep eth0` it was in the `192.168.0.0/16` subnet.
However the `WSL2 X server` firewall inbound rule I wrote whitelisted the `172.16.0.0/12` subnet.
Apparently it can use either private subnet.
I've now updated the instructions to include both.
Because these are private subnets as long as you're only on networks you trust this should be fairly safe to do.

One upside is servers running newer builds of Windows (*after* 18945) then WSL2 servers can be accessed [through localhost](https://docs.microsoft.com/en-us/windows/wsl/compare-versions#accessing-network-applications) on Windows.
This makes testing applications in a browser much easier.