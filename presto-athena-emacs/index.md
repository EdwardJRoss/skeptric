---
date: 2020-03-19 08:44:50+11:00
image: /images/presto_emacs.svg
title: Presto and Athena CLI in Emacs
---

I find having Emacs as a unified programming environment really useful.
When writing an SQL pipeline I can iteratively develop my SQL in emacs, running it against the database.
For a quick and dirty analysis I can copy the output into the .sql file and comment it out.
Then I can copy the SQL into a programming language, parameterise it, and test it without touching the mouse.
So when I started using [Presto](https://prestodb.io/) and AWS's managed alternative [Athena](https://aws.amazon.com/athena/), I needed to integrate it into emacs.

After considering the alternatives I ended up using [Mastering Emacs guide](https://www.masteringemacs.org/article/comint-writing-command-interpreter) to hack together my own comint modes for [Presto](https://github.com/EdwardJRoss/dotfiles/blob/master/emacs.d/lisp/presto.el) and [Athena](https://github.com/EdwardJRoss/dotfiles/blob/master/emacs.d/lisp/athena.el) and it works pretty well.
These use the [Presto CLI](https://prestodb.io/docs/current/installation/cli.html) and [Athena CLI](https://github.com/guardian/athena-cli) to interact with the databases; you will need to have them set up.
To connect with emacs you need to add the files to your `load-path`, and `require` the package (`presto` or `athena`), and configure the path to the binary and the arguments.

```elisp
(add-to-list 'load-path (expand-file-name "lisp" user-emacs-directory))
(require 'presto)
;; Configure path and arguments
(setq presto-cli-file-path "~/bin/presto")
;; Here put whatever arguments you would use to connect through the CLI
(setq presto-cli-arguments '("--server"
                                "localhost:8889"
                                "--catalog"
                                "hive")
```

Then to use it in Emacs you just invoke `M-x run-presto` to start a presto session in a buffer called `*Presto*`.
If you're in a buffer with cursor over some SQL you can send it to be evaluated using `M-x presto-send`.
The rest of this post covers the rationale and the details.


# Ways to integrate SQL into Emacs

The best way to use SQL from Emacs is [sql-mode](https://www.emacswiki.org/emacs/SqlMode); there are [good](https://emacsredux.com/blog/2013/06/13/using-emacs-as-a-database-client/) [guides](https://truongtx.me/2014/08/23/setup-emacs-as-an-sql-database-client) on using it to manage multiple database connection configurations and easily send chunks of SQL from a file to the database.
However I really struggled to add a Presto backend and found the [source documentation](https://github.com/emacs-mirror/emacs/blob/master/lisp/progmodes/sql.el) on adding a product didn't work for me.
More recently someone else has created a [sql-mode backend for presto](https://github.com/kat-co/sql-prestodb) and I will look to migrate to that.

An alternative is [Emacs EDBI](https://github.com/kiwanami/emacs-edbi), which uses Perl's DBI (DataBase Interface) to connect to databases.
This means there is a very large number of databases available and you can connect via ODBC drivers (which your database likely has for your platform).
However installing Perl dependencies is a bit of trouble, and the mode has a detailed workflow and keybindings which I don't find intuitive (especially as an [Emacs evil](https://github.com/emacs-evil/evil) user).
I think it could be a good solution if you spend some time configuring it to be comfortable for you, but I haven't invested in it.

So I resolved to make my own comint mode following [Mastering Emacs comint mode guide](https://www.masteringemacs.org/article/comint-writing-command-interpreter).
Comint mode allows integrating a command line interface through Emacs, which is very flexible and convenient to use.
This worked pretty well with Presto and Emacs

# Setting up Command Line Interfaces

Presto comes with a [good CLI](https://prestodb.io/docs/current/installation/cli.html) written in Java.
You should be able to download it and just run it with Java; try `presto --help` to see all the configuration options.

The story for Athena is much worse; Amazon doesn't give you a nice CLI just a crummy web GUI and an onerous interface through AWS CLI (you have to do your own polling).
There is an [unmaintained Athena CLI](https://github.com/guardian/athena-cli) that works well enough, however it's dependencies are broken.
Make sure you first pip install `cmd2 == 0.8.0` (and maybe `tabulate == 0.8.3`) before trying to run it or it will be broken.
Once it's installed run `athena --help` to see all the configuration options; in particular you will need to set a `--bucket` as a S3 staging bucket for all your query results.
Note that this uses `boto3` to connect to AWS and so you should use [boto3 configuration](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html) for AWS.
In particular the environment variables like `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY` or `AWS_PROFILE` are useful for making sure you can connect with the right role.

Make sure you can run your CLI from the terminal before you try to integrate with emacs.

# Integrating the CLIs into Emacs

Once you can connect and execute queries from the command line you want to encode that into Emacs by defining the path to the program and the arguments.
Here's an example for Presto:

```elisp
(defvar presto-cli-file-path "~/bin/presto"
  "Path to the program used by `run-presto'.")

(defvar presto-cli-arguments '("--server"
                                  "localhost:8889"
                                  "--catalog"
                                  "hive")
  "Commandline arguments to pass to `presto-cli'.")
```

Then we create a `run-presto` function to start the process:

```elisp
(defun run-presto ()
  "Run an inferior instance of `presto-cli' inside Emacs."
  (interactive)
  (let* ((presto-program presto-cli-file-path)
         (buffer (comint-check-proc "Presto")))
    ;; pop to the "*Presto*" buffer if the process is dead, the
    ;; buffer is missing or it's got the wrong mode.
    (pop-to-buffer-same-window
     (if (or buffer (not (derived-mode-p 'presto-mode))
             (comint-check-proc (current-buffer)))
         (get-buffer-create (or buffer "*Presto*"))
       (current-buffer)))
    ;; create the comint process if there is no buffer.
    (unless buffer
      (apply 'make-comint-in-buffer "Presto" buffer
             presto-program '() presto-cli-arguments)
      (presto-mode))))
```

This starts the new process in `presto-mode` which we need to define as a derived mode of `comint-mode`.
See the [Mastering Emacs guide](https://www.masteringemacs.org/article/comint-writing-command-interpreter) or the [source](https://github.com/EdwardJRoss/dotfiles/blob/master/emacs.d/lisp/presto.el) for details.

# Setting the Pager

By default `less` is used as the pager which doesn't work well in Emacs (you'll see `WARNING: terminal is not fully functional` messages).

One solution is to turn off the pager and let Emacs terminal handle all the buffering, using the special environment variables for Presto and Athena CLIs.

```elisp
(setenv "PRESTO_PAGER" "")
(setenv "ATHENA_PAGER" "")
```

Unfortunately I found Emacs comint gets really slow for large results (which terminals handle well), and so if I forgot a limit on a query I could lock up Emacs fetching results.
So I used head to only get the top 1000 results; unfortunately there's no clear sign if the results have been truncated, but for me the tradeoff was worth it.

```elisp
(setenv "PRESTO_PAGER" "head -n 1000")
(setenv "ATHENA_PAGER" "head -n 1000")
```

I will still sometimes find it's slow if I have really long rows; but I don't know how to improve that.

# Sending text to Presto/Athena

Often I like to work in an SQL file and interactively send queries to Presto/Athena.
While I can copy and paste the query into the `*Presto*` or `*Athena*` buffer, it's much nicer to do it with a single command.

It's straightforward to send it using `comint-send-region` (which handles some magic in how newlines are sent to the comint process).

```elisp
(defun presto-send ()
  "Send the current region or paragraph to Presto process."
  (interactive)
  (let ((beg-end (if (use-region-p)
                    (cons (region-beginning) (region-end))
                    (er/paragraph-extents))))
    (comint-send-region "*Presto*" (car beg-end) (cdr beg-end))))
```

The `er/paragraph-extents` function will grab the paragraph under the cursor to send, so if your query is separated by newlines and your cursor is anywhere in it this will do the right thing.

```elisp
(defun er/paragraph-extents ()
  "Return a cons cell with beginning and end of paragraph."
  (save-excursion
    (forward-paragraph)
    (let ((end (point)))
      (backward-paragraph)
      (cons (point) end))))
```

# Quality of life improvements

That's the bare minimum to use it, but I made some changes to make it more usable.

I copied the [Presto keywords](https://prestosql.io/docs/current/language/reserved.html) and added a `font-lock` to highlight them.
This makes it easier to spot spelling mistakes with keywords in the buffer

Presto CLI emits some ANSI control sequences to move the cursor which Emacs doesn't understand, so I followed [oleksandrmanzyuk's guide to filtering them out](https://oleksandrmanzyuk.wordpress.com/2011/11/05/better-emacs-shell-part-i/).

# Pain points

While this works alright there are still a few pain points.

It would be nice to bind `presto-send` to a key, but I don't know what mode to do it with because not all SQL is presto.
The `sql-mode` has a nice way of solving this (and more) with the idea of attaching a process to an editing buffer, but this is pretty hard to implement.
The nicest solution would probably be to integrate with `sql-mode`.

By default long lines are wrapped; I always invoke `toggle-truncate-lines` the first time that happens but that should be baked into the mode by default.

There is some weird behaviour with scrolling through command history.
The first time I invoke `comint-previous-input` with `M-p` the cursor moves back one space.
If I press it again I get the error message `not-at-command-line`.
So I then move the cursor right and invoke `comint-previous-input` again and it retrieves the *2nd to last* input.
I can then get the last input by invoking `comint-next-input` with `M-n`.
I haven't worked out how to work around this so I've got used to this, and tend to send commands rather than using comint more often.
Also history is lost between invocations.

Presto will show the progress of the query as it executes.
In the terminal this will only happen in a single paragraph as the cursor is moved around with ANSI control sequences, and the text is overwritten on update.
However Emacs can't deal with these (in fact these are the ones that we filtered out above), and so we will get a new paragraph for every update.
For a long running query the comint buffer can fill up with these updates.
It should be possible to remove this noise in the comint mode, but I haven't worked out how.

Presto processes can't be killed with `(comint-interrupt-subjob)` (`C-c C-c`), but Athena processes can.
When I make a mistake in Presto I'll delete the `*Presto*` buffer, kill the job in the Presto UI, and then `run-presto` again.
When AWS credentials expire I have to delete the `*Athena*` buffer, update the credentials and `run-athena` again.

Sometimes I want to export the output into another program for sharing.
My workflow is to copy the results into a scratch buffer.
I'll then `regexp-replace` ` +| +` with `,`, and then remove the leading whitespace and delete the line that seperates the header (which works fine unless the row could contain a pipe!).
Then I can copy this CSV into another program.
It would be nice to have an integrated way of doing this (this is one thing [JetBrains Datagrip](https://www.jetbrains.com/datagrip) seems more convenient for).

However it's good enough to be useful for everyday work.