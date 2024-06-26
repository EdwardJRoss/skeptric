---
categories:
- linux
date: '2021-03-10T19:51:01+11:00'
image: /images/pdfview.png
title: Previewing changes to LaTeX documents with inotify
---

Sometimes it's useful to rerun a task whenever a file changes; whether that's a linter or a test suite, or a preview.
I recently wanted to recompile a TeX file to PDF whenever I saved a change, and it was easy with inotify, [using instructions from superuser](https://superuser.com/questions/181517/how-to-execute-a-command-whenever-a-file-changes).

To install inotify on a Debian derivative you can use `sudo apt install inotify-tools`.

Then you can set it to run a command whenever a file is done saving.
In my case I have a Makefile to convert the TeX file into a PDF so I can run the following command:

```
while inotifywait -e close_write Edward_Ross.tex; do make; done
```

I can view the PDF in Emacs with the excellent [pdf-tools](https://github.com/politza/pdf-tools); and by setting the buffer to automatically update on change with `M-x auto-revert-buffer` whenever I save my TeX file the PDF is generated and the buffer updates with the changes.
It even seems to roughly keep my position.
While it's not perfect, and for longer documents the refresh time will be large, it's a convenient way to quickly preview small changes to a TeX or LaTeX document.