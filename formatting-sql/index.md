---
categories:
- sql
- python
- emacs
date: '2021-01-30T08:00:00+11:00'
image: /images/sqlformat.png
title: Offline SQL Formatting with sqlformat
---

It's polite to format your SQL before you share it around.
You want to be able to do it in context, and not upload your private SQL to some random website.
The `sqlformat` command of the Python package `sqlparse` is a great tool for the job.

You can install `sqlformat` in Debian derivatives such as Ubuntu with `sudo apt install sqlformat`.
Alternatively with any system with Python you can install it via `pip install sqlparse`, just make sure you have the binary in your path (e.g. if using `pip install --user` then `~/.local/bin/` should be in your path).

The formatting is provided by the Python library [sqlparse](https://sqlparse.readthedocs.io/en/latest/intro/), and you can see an example of how it works at [sqlformat.org](https://sqlformat.org/).
The instructions are at `sqlformat --help` and there's lots of arguments to choose how to format keywords, identifiers, alignment, comma placement and wrapping.
Here's a simple command that takes the contents of query.sql, converts the keywords to upper case, reindents the statements and puts space around operators, and outputs to formatted.sql:

```
sqlformat -k upper -r -s -o formatted.sql query.sql
```

The command line is useful for batch changing SQL files, but often I want to change an SQL query inside a Python or R file.
This works great, even if you're using parameters, f-strings in Python or glue in R, it seems to do a good job.
It's straightforward to run on a region of SQL in Vim (or Emacs Evil) [using a filter command](https://vim.fandom.com/wiki/Append_output_of_an_external_command#Using_a_filter_command), or in Emacs [by executing a shell command](https://www.masteringemacs.org/article/executing-shell-commands-emacs), just make sure you pass `-` as the file so it reads from stdin.

For Emacs I've written a small function to SQL format a region; being careful not to clobber the final newline of the region.
A nice improvement would be to customise the arguments.

```elisp
(defun remove-trailing-newline (point)
  (if (= (char-before point) ?\n)
      (- point 1)
    point))

(defun sql-format (start end)
  "Formats the selected sql `sqlformat'"
  (interactive "r")
  (shell-command-on-region
   ;; beginning and end of buffer
   start
   (remove-trailing-newline end)
   ;; command and parameters
   "sqlformat -k upper -r -s -"
   ;; output buffer
   (current-buffer)
   ;; replace?
   t
   ;; name of the error buffer
   "*Sqlformat Error Buffer*"
   ;; show error buffer?
   t))
```