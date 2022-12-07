---
categories:
- emacs
- blog
date: '2021-11-22T08:00:00+11:00'
image: /images/hugo_readdir_error.png
title: Hugo Readdir Error with Emacs
---

Every now and then when previewing Hugo (via `hugo serve`) as I'm editing it in Emacs I'll get a strange error like:

```
ERROR 2021/10/18 19:36:03 process: readAndProcessContent: walk:
Readdir: decorate: lstat
/home/user/skeptric/content/user@machine.2139:12345 no such file or directory
```

Often I can work around it by editing and saving the file I've been editing again, but I'll have to restart the Hugo process.
However the underlying cause is *lockfiles* in Emacs, and the easiest fix is to run the elisp (in [24.3 and above](https://stackoverflow.com/questions/5738170/why-does-emacs-create-temporary-symbolic-links-for-modified-files)):

```elisp
(setq create-lockfiles nil)
```

# Explanation

If I got the above error when editing a file called `about.md` I'd normally be able to find a symlink like `.#about.md -> user@machine.2761:12345`.
This turns out to be emacs system for [file locks](https://www.gnu.org/software/emacs/manual/html_node/elisp/File-Locks.html#File-Locks).
The `.#<filename>` is symlinked to `<user>@<host>.<pid>:<time since boot>`.
Then when another user tries to to open the file in emacs it will then complain about the file being locked.
You can [turn this off](https://www.gnu.org/software/emacs/manual/html_node/emacs/Interlocking.html) by setting `create-lockfiles` to `nil`.

Hugo tries to follow these symlinks and realising they're not going to a real file complains.
Unfortunately I can't find any way to disable this behaviour of Hugo, and since I'm not in a multiuser environment it seems the only solution is to turn off Emacs lockfiles.