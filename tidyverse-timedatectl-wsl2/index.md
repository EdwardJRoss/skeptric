---
categories:
- R
- tidyverse
- debug
date: '2021-06-12T22:38:09+10:00'
image: /images/tidyverse_timedatectl.png
title: Installing Tidyverse in WSL without Timedatectl Status 1 Issue
---

When I tried to install tidyverse in WSL2 I ran into issues with timedatectl and xml2.
The simple solution is:

```
# Assuming Debian derivatives
sudo apt-get install libxml2-dev
# Modify TZ to whatever your timeozne is
TZ="Australia/Sydney" R -e 'install.packages("tidyverse")'
```

# What happens

When I try to install tidyverse I get this error:

```
> install.packages('tidyverse')

ERROR: configuration failed for package ‘xml2’
System has not been booted with systemd as init system (PID 1). Can't operate.
Failed to create bus connection: Host is down
Warning in system("timedatectl", intern = TRUE) :
  running command 'timedatectl' had status 1
Error in loadNamespace(j <- i[[1L]], c(lib.loc, .libPaths()), versionCheck = vI[[j]]) :
  namespace ‘xml2’ 1.3.1 is already loaded, but >= 1.3.2 is required
Calls: <Anonymous> ... namespaceImportFrom -> asNamespace -> loadNamespace
Execution halted
ERROR: lazy loading failed for package ‘tidyverse’
```

Indeed `timedatectl` fails on WSL2 and [there's an open issue about it](https://github.com/microsoft/WSL/issues/6417).

```
~ $ timedatectl
System has not been booted with systemd as init system (PID 1). Can't operate.
Failed to create bus connection: Host is down
```

So it seems that `timedatectl` is being called in xml2 to get the timezone.
While I find this a bit offputing, you can [specify the timezone](https://r.789695.n4.nabble.com/Sys-timezone-fails-on-Linux-under-Microsoft-WSL-td4768543.html) with the TZ environment variable, and then it doesn't call `timedatectl`.

So putting this together gets to our original answer:

```
TZ="Australia/Sydney" R -e 'install.packages("tidyverse")'
```