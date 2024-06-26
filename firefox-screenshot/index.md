---
categories:
- emacs
- linux
date: '2021-04-14T21:08:38+10:00'
image: /images/firefox-screenshot.png
title: Taking Screenshots in Firefox
---

I find taking screenshots in Linux a bit painful.
My current way is to use GIMP to create an image from a screenshot, but it's a bit slow to startup and interrupts my flow.
I've had trouble installing Shutter which I haven't worked through yet.
However I've just found out that Firefox has a [way to take screenshots](https://support.mozilla.org/en-US/kb/firefox-screenshots).
All you need to do is press Control-Shift-S and then it brings up a selector where you can pick an element, or a region (like an improved version of Windows Snipping tool).

![Taking Screenshots in Firefox](/images/firefox-screenshot.png)

You can then choose to download it, or copy it to the clipboard.
If you copy it to the clipboard you can actually save it to a file using xclip:

```sh
xclip -selection clipboard -t image/png > screenshot.png
```

To take this one step further my most common use for this is pasting images into my this website.
I created a hacky function in Emacs to make this easy (baking in the fact that for this hugo site images are stored in `../../static/images/` relative to the posts).
It prompts for a name and saves the file in the right location as `{name}.png` and creates a Markdown image link in the current buffer.
It doesn't handle all the way things could go wrong (special characters in the name, clipboard is empty, file exists, etc.), but is a useful starting point.

```elisp
(defun er/hugo-save-clipboard-image (name)
  "Saves image from clipboard to NAME."
  (interactive "sFile name: ")
  (let ((outfile (concat "/images/" name ".png")))
        (call-process-shell-command
         (concat "xclip -selection clipboard -t image/png -o >../../static" outfile)
         "/dev/null"
         0)
        (save-excursion
          (insert (concat "![](" outfile ")")))
        (forward-char)
        ))
```
