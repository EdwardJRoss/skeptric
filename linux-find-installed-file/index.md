---
categories:
- programming
date: '2020-10-19T21:49:15+11:00'
image: /images/dpkg.png
title: Finding Files Installed in Ubuntu and Debian
---

My bashrc file sources the git prompt helper to show the branch I'm on in the prompt.
Unfortunately it's quite old and was pointing to the wrong file, how do I find where it is?

```
dpkg -L git | grep prompt
```

Debian and its derivatives such as Ubuntu you can use `apt` to manage packages (e.g. `apt upgrade`, `apt install`).
However `apt` is just a thin layer over `dpkg` that does useful things like resolving dependencies and downloading files.
If you want to get low level with packages in Debian derivative distriubutions, look into `dpkg`.

From the man page `--listfiles` or equivalently `-L` lists all files installed from a package.
So `dpkg -L git` shows all files installed in the git package, including the prompt.
We can then use `grep` to filter the results.

In Arch Linux it's `pacman -Ql git`.
For systems such as Fedora using yum it's `repoquery -l git`.