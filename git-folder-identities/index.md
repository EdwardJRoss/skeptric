---
categories:
- linux
- programming
date: '2020-07-22T08:43:14+10:00'
image: /images/git_identities.svg
title: Git Folder Identities
---

Sometimes you want a different [git configuration](https://git-scm.com/book/en/v2/Customizing-Git-Git-Configuration) in different contexts.
For example you might want different author information, or to exclude files for only *some* kinds of projects, or to have a specific template for certain kinds of projects.

The easiest way to do this consistently is with a `includeIf` statement.
For example to have custom options for any git repository under a folder called `apache` add this to the bottom of your `~/.gitconfig`.

```
[includeIf "gitdir:apache/]
	path = .gitconfig_apache
```

Then put any custom configuration options in `~/.gitconfig_apache`.
Then if you have a git folder in `~/src/apache/code/my-repo` then any configuration in `~/.gitconfig_apache` will be applied, but it won't by if you're in `~/src/random/my-repo` (since `apache/` is not in the path to the repository root).

If you just want the configuration to apply to a few specific repositories you can add local configuration in each repository in `.git/config`.
Between these two methods you should be able to get the context dependent configuration you want.

# The details of git configuration

Configuration is obtained by reading through various config files in order from top to bottom, where later assignments overwrite previous; so for example a variable in a local configuration will overwrite a variable in a global configuation.

```
system configuration:    $(prefix)/etc/gitconfig
global configuration:    ~/.gitconfig
local configuration:     (repo)/.git/config
Environment variables
Command line
```

I'm skipping over [some details](https://git-scm.com/docs/git-config) including [worktree configuration](https://stackoverflow.com/questions/31935776/what-would-i-use-git-worktree-for), but this captures most of it.
One other handy thing to know is the local configuration can also be specified to be a different file with the `GIT_CONFIG` environment variable or the `-f/--file` command line argument.

This gives you plenty of ways to set your configuration; especially when you combine this with `includeIf` (and the unconditional `include`) which inserts the referenced configuration file at that point based on the location of the `gitdir` or the name of a branch with `onbranch`.

You could actually implement the `includeIf` behaviour by setting environment variables [based on your working directory](https://makandracards.com/makandra/19549-how-to-set-git-user-and-email-per-directory), but that's a bit flaky and won't work if you use some tool outside the shell you configured it in.
The `includeIf` method is [pretty common](https://www.kevinkuszyk.com/2018/12/10/git-tips-6-using-git-with-multiple-email-addresses/) and seems to work well.

The `gitdir` (and `onbranch`) in `includeIf` uses a [gitignore](https://git-scm.com/docs/gitignore) style syntax so you could even do funky things like `gitdir:~/src/**/apache/*/.git` to get any repositories under `~/src` one folder below an `apache` directory.
I don't know why you would, but you certainly could.

One final gotcha; `gitdir` only works if you're in a git directory.
So you can sort-of think of `gitdir:apache/` as equivalent to `gitdir:**/apache/**/.git`, and the configuration won't apply in `~/src/apache/` if that's not a git directory.
It rarely comes up, but can be confusing when debugging.