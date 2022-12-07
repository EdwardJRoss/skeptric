---
categories:
- emacs
- programming
date: 2020-03-20 22:31:01+11:00
image: /images/dumbjump.png
title: Using emacs dumb-jump with evil
---

[Dumb-jump](https://github.com/jacktasia/dumb-jump) is a fantastic emacs package for code navigation.
It jumps to the definition of a function/class/variable by searching for regular expressions that look like a definition using [ag](https://github.com/ggreer/the_silver_searcher), [ripgrep](https://github.com/BurntSushi/ripgrep) or git-grep/grep.
Because it is so simple it works in over 40 languages (including oddities like SQL, LaTeX and Bash) and is easy to extend.
While it is slower and less accurate than [ctags](https://en.wikipedia.org/wiki/Ctags), for medium sized projects it's fast enough and requiring no setup makes it much more useful in practice.
You can even configure a simple `.dumbjump` file (similar to a `.gitignore`) to exclude directories to make it faster, or include other directories to jump between packages.

I use it with [Evil mode](https://github.com/emacs-evil/evil) (vim emulation) and bind `dump-jump-go` to `C-]`.
Unfortunately it doesn't play well with the Evil jump list and when I invoke `evil-jump-backward` with `C-o` it jumps back to the wrong place.
While I could invoke `dumb-jump-back`, I like being able to use `C-o` for a `dumb-jump` or a search in emacs.
This can be done by advising the core jump function in dumb-jump to push the position to evil's jump list.
There's probably a prettier way but this does the trick:

```elisp
;; Preserve jump list in evil
(defun evil-set-jump-args (&rest ns) (evil-set-jump))
(advice-add 'dumb-jump-goto-file-line :before #'evil-set-jump-args)
```

If you're ever working in Vim proper [dim-jump](https://codelearn.me/2018/12/02/vim-dumb-jump.html) is a substitute for dumb-jump.