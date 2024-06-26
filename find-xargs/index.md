---
categories:
- programming
date: '2020-12-12T20:54:35+11:00'
image: /images/man_xargs.png
title: Using find and xargs
---

Sometimes you want to feed a bunch of files to a program, and this is often easily done with find and xargs.

Suppose you have an executable `doit` that you want to execute on all Python files in `src/`; you can do this directly with find:

```
find src/ -name '*.py' -exec doit {} \;
```

You can use xargs for this as well; but if there's a chance that a path could contain a space somewhere it's best to use `-print0` with find and `-0` with xargs to separate all arguments with nulls (rather than spaces):

```
find src/ -name '*.py' -print0 | xargs -0 -n1 doit
```

The nice thing about using find with `-exec` is you can put the placeholder `{}` in different places; for example if you wanted to *run* a bunch of bash scripts passing the argument `foo` you could use:

```
find src/ -name '*.sh' -exec bash {} foo
```

With `xargs` you can do the same thing using `-I` to specify a replace string:

```
find src/ -name '*.sh -print0 | xargs -I{} -0 -n1 bash {} foo
```

An advantage of xargs is you can pass multiple arguments at a time.
When running Python or Java scripts there's quite a bit of runtime overhead in starting up a program, so it can be significantly faster if they can process multiple files themselves.
By default it passes all the arguments:

```
find src/ -name '*.py' -print0 | xargs -0 doit
```

A very useful feature of xargs is it can run multiple scripts in parallel.
This is a really handy batching mechanism, especially for I/O bound operations like Pythons [multiprocessing](/multiprocess-download) and [futures](/multiprocessing-future).
For example to run 4 threads in parallel, passing 5 arguments at a time (so 20 files get processed at once) you could run:

```
find src/ -name '*py` -print0 | xargs -0 -P4 -n5 doit
```

Of course there are things where find and xargs won't be enough and you want to use a programming language or a framework.
But they're super useful for quickly processing files.

Note that if you're looking for files containing a certain piece of test you can use `grep`, `ag` or `rg` with xargs as well:

```
grep -r -l -Z --include '*.pyi' 'import Path'  src/ | xargs -0 echo
ag -l -0 --python 'import Path' src/ | xargs -0 echo
rg -l -0 --iglob '*.py' 'import Path' src/ | xargs -0 echo
```