---
categories:
- emacs
- linux
date: '2020-07-23T08:00:00+10:00'
image: /images/dotfiles.png
title: Customising Portable Dotfiles
---

I keep my personal configuration files in a public [dotfiles](https://github.com/EdwardJRoss/dotfiles) repository.
This means that whenever I'm on a new machine it's very easy to get comfortable in a new environment.
However I find I often need machine specific configuration, so I provide ways to override them with local configuration.

When I get to a new machine I'll pretty quickly want some of my usual configuration (although I don't *need* it).
I can clone or download a zipfile of my dotfiles and then install it via some symlinks via a [bootstrap bash script](https://github.com/EdwardJRoss/dotfiles/blob/master/bootstrap.sh).
There are better tools for managing dotfiles like [rcm](https://github.com/thoughtbot/rcm) but they have dependencies that may not be easy to install (especially on oddball systems like Cygwin).

Because they're symlinks I can edit the files, and push any changes back upstream.
The problem is that sometimes I want *private* changes that I don't want to push upstream.
This is solved with the `.local` pattern; extend the configuration with a configuration file ending in `.local` if it exists.
So for example a `.bashrc` is extended by a `.bashrc.local`, and a `.gitconfig` is extended by a `.gitconfig.local`.
These local files are not in public version control, and I can add them as necessary to any machine.

## Git

For git configuration you can use the [Include](/git-folder-identities) statement to insert configuration from a custom file, and nothing happens if the file doesn't exist.
So your gitconfig file looks like:

```
...
[include]
  path = ~/.gitconfig.local
```

## Bash

In your bashrc you can add configuration with source, but you have to check the file exists to let it work when there is no local configuration.

```
if [[ -e ~/.bashrc.local ]]; then
    source ~/.bashrc.local
fi
```

## Emacs

Emacs also tracks variables that have been customised through the [easy customisation interface](https://www.gnu.org/software/emacs/manual/html_node/emacs/Easy-Customization.html#Easy-Customization).
Rather than having this litter the version controlled init file it makes sense to put these into a local file too; they can be migrated to the general configuration as desired.
The `.local` convention is a little less clear here because I store the config in `.emacs.d/init.el`; so I call the custom file `init.local.el`.

```
;; Put/save customisations through customize in a separate file
(defconst custom-file (expand-file-name "init.local.el" user-emacs-directory))
(unless (file-exists-p custom-file)
  (write-region "" nil custom-file))
(load custom-file)
```

## Everything else

This pattern works across a large number of configuration files, but the syntax is slightly different each time.
The [thoughtbot dotfiles](https://github.com/thoughtbot/dotfiles) have great examples showing how to implement it for zshell, tmux, and psql .

Happy configuring!