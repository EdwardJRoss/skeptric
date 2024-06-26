---
categories:
- programming
- python
date: '2020-12-15T20:50:53+11:00'
image: /images/pipenv_error.png
title: Managing Python Versions with asdf
---

I was recently trying to run a pipenv script, but it gave an error that it required Python 3.7 which wasn't installed.
Unfortunately I was on Ubuntu 20.04 which has Python 3.8 as default, and no access to earlier versions in the repositories.
However pipenv gave a useful hint; pyenv and asdf not found.

The [asdf](https://asdf-vm.com) tool allows you to configure multiple versions of applications in common interactive shells (Bash, Zsh, and Fish).
In this case it could handle the installation of any version of Python and pipenv could use it to access that version.
But the tool is much more versatile than that and can work with a [wide variety of languages](https://asdf-vm.com/#/plugins-all) including R, Node.js and Java.
It gives an easy way to install and set the version; which seems much better than fussing about with JAVA_HOME.

It is very easy to install asdf.
There's a very good [guide to installing](https://asdf-vm.com/#/core-manage-asdf) that lets you pick your method.
You can then install a plugin like `asdf plugin add python`, and you're ready to go.
This was enough for pipenv to detect the version and prompt installing it.
It seemed much easier to setup than pyenv, while being much more versatile since pyenv is Python specific.

You will need any compile time dependencies installed before you install the plugin.
In my case I got an error `No module named '_ctypes'` because [I was missing libffi](https://stackoverflow.com/a/48045929).
I could resolve this by `apt install libffi_dev` and then reinstalling the plugin; `asdf uninstall python 3.7.9` and then `asdf install python 3.7.9`.
Ultimately this shows that it's less reliable than using a container, where you can bundle all the dependencies of the right version, but asdf is a convenient low-overhead way of managing multiple versions for a programming lanugage.