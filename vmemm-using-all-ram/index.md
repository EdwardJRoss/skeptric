---
categories:
- wsl
date: '2020-11-13T17:05:50+11:00'
image: /images/vmemm_ram.png
title: Why is Vmemm Using All My Memory?
---

My Windows laptop was halting to a crawl; I was waiting seconds to switch windows and even typing took a couple of seconds to respond.
I opened the task manager by hitting Ctrl-Shift-Esc and saw that Vmemm was using >95% of my memory.
What the heck is Vmemm and how can I stop it using all my memory?

Vmemm is the process associated with virtual machines on Windows.
I'm using WSL2 and Docker (through WSL2), and so all their memory appears on Vmemm.
But I had no Docker containers running, and `free` in WSL2 said almost all the memory was available.

It seems that when Linux under WSL2 frees memory the cached memory isn't made available back to Windows.
So whenever I ran heavy docker containers the memory wouldn't be available back in Windows anymore.

There's a long [WSL issue on Github](https://github.com/microsoft/WSL/issues/4166) addressing this; there's no solution but there's a workaround.
You can always free up the memory by restarting WSL2; using `wsl --shutdown` (e.g. using Command Prompt or Powershell) and then starting up the process again.
But for a longer term solution you can *limit* the amount of memory available to WSL using `.wslconfig`.

The file at `C:/Users/<user name>/.wslconfig` gives configuration options for WSL, and in particular you can set a maximum memory and number of processors available to WSL.
The full [configuration syntax is in the documentation](https://docs.microsoft.com/en-us/windows/wsl/wsl-config#configure-global-options-with-wslconfig).
In my case 4GB was a good tradeoff so I updated the file as follows:


```
[wsl2]
memory=4GB # Limits VM memory in WSL 2 to 4 GB
processors=2 # Makes the WSL 2 VM use two virtual processors
```

Limiting the memory available to WSL2 in the wslconfig at least stops it from bringing my whole machine to a crawl.