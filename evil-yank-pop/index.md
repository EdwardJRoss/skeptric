---
categories:
- emacs
date: '2021-04-19T20:37:53+10:00'
image: /images/vim_registers.png
title: Pasting text from long ago in Emacs and Vim
---

I use Vim keybindings in Emacs through [Evil Mode](https://github.com/emacs-evil/evil) and [Evil Collection](https://github.com/emacs-evil/evil-collection/).
Often I'll copy something, make some edits, and then want to paste the text.
The problem is that the edits changed what was on the register and I want to recover that text.

In Vim (and Evil mode) I can type `:reg` to see what's in the registers, find the value I want to paste, commit the register to memory (for example `8`), exit and then paste that register (`"8p` from normal mode, `C-r 8` from insert mode).
This is a bit painful and breaks the flow.
If I'm copying from in Vim/Emacs and thinking ahead I can copy to a certain register (say a `"y<motion>`), and then paste back with that same register, but I don't find that at all natural.
To find out more about Vim registers either type `:h registers` or read [Brian Storti's article on it](https://www.brianstorti.com/vim-registers/).

Emacs has a notion of a `kill-ring`, which contains all the recently killed (Emacs-speak for cut) text.
You can then `yank` (Emacs-speak for paste) the text from the top of the ring (bound to `C-y`).
But then if you wanted something further back you can use `yank-pop` (bound to `M-y`) to go back through the kill ring until you find the text that you want.
Evil supports this with `evil-paste-pop` (overwriting the `M-y` binding by default); you can then paste with `p` or `P` (`evil-paste-after` or `evil-paste-before`) and then continue pressing `M-y` until you get back to the text you want.
Vim (but not Evil) also supports this [through the redo-register](https://stackoverflow.com/a/17014279); after you paste from the first delete register with `"1p` you can scroll through the registers with `u.`.

I still find the Vim and Evil mechanisms a bit clunky because it requires switching keybindings.
But the [Counsel completions](https://github.com/abo-abo/swiper#counsel) for Emacs contains `counsel-yank-pop` which opens the completions in an Ivy minibuffer.
This is great because you can select a completion either by navigating with up and down arrows (or `C-p` and `C-n` keys), or by typing a part of the text you want and then selecting it to paste with `Enter`.
I find this a much easier way to retrieve text from long ago.
I've bound this to `gp` in normal mode and `Ctrl-Shift-r` in insert mode.

```elisp
(use-package counsel
  :demand t
  :bind (:map evil-normal-state-map
         ("gp" . 'counsel-yank-pop)
         :map evil-insert-state-map
         ("C-S-r" . 'counsel-yank-pop))
   ;... Rest of configuration
  )
```