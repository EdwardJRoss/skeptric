---
categories:
- emacs
- vscode
date: '2020-05-09T08:50:50+10:00'
image: /images/vscode.svg
title: Considering VS Code from Emacs
---

I've been using Emacs as my primary editor for around 5 years now (after 4 years of Vim).
I'm very comfortable in it, having spent a long time configuring my [init.el](https://github.com/EdwardJRoss/dotfiles/blob/master/emacs.d/init.el).
But once in a while I'm slowed down by some strange issue, so I'm going to put aside my sunk configuration costs and have a look at using VS Code.

# On Emacs

I recently read a LWN article on [Making Emacs Popular Again](https://lwn.net/SubscriberLink/819452/1480c3a59d3d9093/) (and the corresponding [HN thread](https://news.ycombinator.com/item?id=23107123)).
It's very hard to bring in new libraries and external code to Emacs because of their strict requirements of GPL3+ (so even [Qt's](https://www.qt.io/) GPL3 [isn't sufficient](https://lwn.net/ml/emacs-devel/E1jPGhC-0003i1-W4@fencepost.gnu.org/)) and requirement of code [copyright assignation to FSF](https://www.gnu.org/software/emacs/manual/html_node/emacs/Copyright-Assignment.html), and even if the primary contributors are fine to sign away their copyright it is difficult for established projects with multiple contributors and a barrier to entry for new contributions.
While I am sympathetic to free software the GNU approach puts a high barrier on entry.
The primary reason Emacs has seen a resurgence is the ability to install third-party packages from [MELPA](https://melpa.org/), which contains packages with the kind of features in modern IDEs.

That being said there are many great features built into Emacs like [remote editing with Tramp](https://www.gnu.org/software/emacs/manual/html_node/tramp/Quick-Start-Guide.html), ability to use [shells](https://www.masteringemacs.org/article/running-shells-in-emacs-overview), working in [directories](https://www.gnu.org/software/emacs/manual/html_node/emacs/Dired.html), [process management](https://www.masteringemacs.org/article/displaying-interacting-processes-proced) and much more.
There are lots of inbuilt functionality from how things are displayed to obscure gems like RFC1345 input mode for typing unicode characters.
There are some useful packaged that can be installed directly via [Elpa](https://elpa.gnu.org/) (the official Emacs package repository where everything is owned by FSF) like [Org Mode](https://orgmode.org/).

The best thing about emacs is everything is very customisable with [Hooks](https://www.gnu.org/software/emacs/manual/html_node/emacs/Hooks.html) and writing functions in [Elisp](https://www.gnu.org/software/emacs/manual/html_node/elisp/).
This is great for making little tweaks, but for large groups of functionality it requires much more.

With MELPA packages Emacs can quickly become a powerful environment for text editing.
I move around text using Vim keybinding with [Evil](https://github.com/emacs-evil/evil), [Evil Collection](https://github.com/emacs-evil/evil-collection/) and [Evil God State](https://github.com/gridaphobe/evil-god-state/) using my own custom commands to navigate between windows or launch applications.
I use [Magit](https://magit.vc/) for interfacing with Git, which frequently saves me from commiting inadvertent changes (and reaches the deeper features of git better than the generic [VC mode](https://www.gnu.org/software/emacs/manual/html_node/emacs/Version-Control.html)).
I quickly navigate menus with [Ivy](https://github.com/abo-abo/swiper), jump through code with [dumbjump](https://github.com/jacktasia/dumb-jump) and complete code with [Company Mode](https://company-mode.github.io/).
I can use [Elpy](https://github.com/jorgenschaefer/elpy) for Python, [ESS](https://ess.r-project.org/) for R, and I've been hearing good things about using [Lanugage Server Protocol](https://emacs-lsp.github.io/lsp-mode/) which provides backends to lots of languages not well supported in Emacs (like Java).

However this requires a lot of time to understand what packages to use and how to configure them.
It's good that it is often easy to customise them, because often it's inconsistent between packages, wbut sometimes it requires a lot of customisation to just make it workable.
And then sometimes there are painfully weird interactions.
Company mode sometimes makes it *really* slow to type in a comint buffer (like a shell), and it took me a long time to figure out that was the issue.
Now I just live without completion in comint buffers, which is a loss.
When I run a command that starts a process it might sometimes run it remotely over TRAMP when I launch it in a local buffer.

All the time I spend configuring and debugging Emacs is time I'm not spending doing something more productive, so I want to see if there's a better option now.

# Alternatives to Emacs

I'm already using Vim for small edits, Jupyter Notebooks for exploratory analysis and sometimes RStudio when I want to use some HTML R features, or for some R Markdown (because [Emacs polymode](https://github.com/polymode/polymode) is a little buggy around the edges).
But none of these could be a full time alternative for me; Vim is less of an IDE than Emacs and the others only work for limited languages and tasks.
It seems like the most popular alternatives from my colleagues is Jetbrains' products and VSCode (with Sublime and Atom seeming to lose mindshare), and they both look viable.

From the outside VS Code looks promising.
It can interact with [WSL](https://code.visualstudio.com/docs/remote/wsl), [Containers](https://code.visualstudio.com/docs/containers/overview) and over [SSH](https://code.visualstudio.com/docs/remote/ssh).
It has a [Vim Emulation Mode](https://github.com/VSCodeVim/Vim) which looks like it has enough features to be useful (though not as many as Evil).
I've seen you can start a terminal from inside VSCode.
It's got documentation that focuses on usage, which is a big advantage over Emacs (which has complete documentation but requires serious dedication to read through).

I'm concerned about Electron memory usage; Slack already uses a lot of memory.
I'm also concerned how it will do for general tasks like editing text files, SQL and whatever else I come across.
But it seems to have a lot of plugins so we'll see what it can do.

# Other people's experiences

A Google search shows a lot of people have switched between the two, and I thought it may be useful to get their perspectives.

The best article I found was [AdmiralBumbleBee's comparison](https://www.admiralbumblebee.com/programming/2020/01/04/Six-months-VS-Code.html), which highlights that window management, navigation, and configuration is better in Emacs, but terminals are better in VS Code.
They also mention having trouble with magit and syncing their `init.el` file, but those are things I like about Emacs.
This resonates with a [Reddit post](https://www.reddit.com/r/emacs/comments/8h1cxa/any_long_time_emacs_users_tried_vscode/dyhf84a/) that says VS Code is viable but easy to configure, limits how to split window and only working on a "per-project" basis (and I find this frustrating when I work in R Studio).

[Hadi Timachi](https://hadi.timachi.com/2019/12/07/Why_I_switched_from_VScode_to_Emacs) moved from Emacs to VS Code and back again, but his issues don't seem compelling.
It seems worth a try.

# Making Emacs Better

Reflecting on the article trying to make Emacs more attractive to new users, what Emacs really needs is more contributors that can help the project.
From core contributors who are very friendly to the community like Eli Zaretskii, to people who write about Emacs for general people like [Magnar Sveen of Emacs Rocks](http://emacsrocks.com/) and [Mickey Peterson of Mastering Emacs](https://www.masteringemacs.org), to prolific package builders like [Abo-Abo](https://github.com/abo-abo/) and [tarsius](https://github.com/tarsius).
While reducing friction and pain point for newcomers will definitely help, Emacs should really focus on what makes these people like Emacs, what makes it different from its competitors (though new competitors like VS Code are changing the gaps).