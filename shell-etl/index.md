---
categories:
- data
- linux
date: '2020-03-23T21:48:51+11:00'
image: /images/history_count.png
title: Data Transformations in the Shell
---

There are many great tools for filtering, transforming and aggregating data like SQL, R dplyr and Python Pandas (not to mention [Excel](/using-excel.mmark)).
But sometimes when I'm working on a remote server I want to quickly extract some information from a file without switching to one of these environments.
The standard UNIX tools like `uniq`, `sort`, `sed` and `awk` can do blazing fast transformations on text files that don't fit in memory and are easy to chain together.
I've used it for getting summaries of hundreds of GB of compressed CSVs, and it works pretty well!
They're slightly awkward to work with, have limited features, and they will break when the data doesn't quite match the standard form, but for a quick diagnostic summary they can be really useful.

I'll show how to get the most frequent commands from your bash history and their frequency.
But the same types of techniques are useful for tallying many types of files, and once you get practiced they're really quick to write.
The final output will look like this:


```bash
> cut -d' ' -f1 ~/.bash_history | \
  sort | \
  uniq -c | \
  sort -nr | \
  sed -e 's/^ *//' -e 's/ /,/g' | \
  awk -F',' -v total="$total" \
  'BEGIN {OFS=","} 
  {cumsum+=$1; print $2, $1, $1/total * 100 "%", cumsum/total * 100 "%"}' | \
  head -n5 | \
  column -ts,

ls   2470  20.7703%  20.7703%
cd   1509  12.6892%  33.4595%
git  794   6.67676%  40.1362%
vim  468   3.93542%  44.0716%
aws  464   3.90178%  47.9734%
```

# Building the pipeline


Suppose that you want to know the most frequent commands that you type at the shell.
If you use bash your recent commands will be stored in `~/.bash_history` (and you can configure it to store [all your history](https://superuser.com/questions/137438/how-to-unlimited-bash-shell-history)).
You can glimpse the most recent commands with `tail` which prints the first few rows (similar to `limit` in most SQL engines).

```bash
> tail -n5 ~/.bash_history
ls
ls ~/tmp
rm ~/tmp/*
ls ~/tmp
df -h
```

Let's try to extract the commands and not the arguments; we use `cut` a little like `select` in SQL, but we have to specify how the columns are separated.
We set the delimiter (`-d`) to a ' ' and the fields to select is the first field `-f1`.
Note that in bash you can specify a tab separated file with `-d$'\t'`.

```bash
> cut -d' ' -f1 ~/.bash_history | tail -n5
ls
ls
rm
ls
df
```

Now we want to count the commands; we can use `uniq -c` which counts repeated lines

```bash
> cut -d' ' -f1 ~/.bash_history | tail -n5 | uniq -c
      2 ls
      1 rm
      1 ls
      1 df
```

This is great for counting streaks, but to count the command we need to put repeated instances of a command together with `sort`:

```bash
> cut -d' ' -f1 ~/.bash_history | tail -n5 | sort | uniq -c
      1 df
      3 ls
      1 rm
```

Finally we want to put them in order by `sort`ing them in reverse (`-r`) numerical (`-n`) order:

```bash
> cut -d' ' -f1 ~/.bash_history | tail -n5 | sort | uniq -c | sort -nr
      3 ls
      1 df
      1 rm
```

Let's use a bit of `sed` magic to make the formatting a nice CSV that we could use in another tool (assuming there are no commas in the data itself!)
Also we'll start breaking the lines to make it more readable.

```bash
> cut -d' ' -f1 ~/.bash_history | \
  tail -n5 | \
  sort | \
  uniq -c | \
  sort -nr | \
  sed -e 's/^ *//' -e 's/ /,/g'

3,ls
1,df
1,rm
```

We can now ://superuser.com/questions/137438/how-to-unlimited-bash-shell-historyltake the `tail` out of the pipe and see our most frequent commands using `head` to only get the top ones:

```bash
> cut -d' ' -f1 ~/.bash_history | \
  sort | \
  uniq -c | \
  sort -nr | \
  sed -e 's/^ *//' -e 's/ /,/g' | \
  head -n5

2470,ls
1509,cd
794,git
468,vim
464,aws
```

Clearly I spend a bit of time navigating directories, git repositories and using AWS.
But how much of my whole shell activity does this represent?
We can get the total lines counting lines with the word count utility: `wc -l` (the count might change under us but it doesn't matter too much).
Note that `wc` prints out the name of the file as well so to just get the number we'll have to select it with `cut`.

```bash
> wc -l ~/.bash_history
11890 ~/.bash_history
> export total=$(wc -l ~/.bash_history | cut -d' ' -f1); echo $total
11891
```

We can now use `awk` to calculate the percentage of time we spend on each command, and finally use `column` to make the output aligned for reading.

```bash
> cut -d' ' -f1 ~/.bash_history | \
  sort | \
  uniq -c | \
  sort -nr | \
  sed -e 's/^ *//' -e 's/ /,/g' | \
  awk -F',' -v total="$total" \
  'BEGIN {OFS=","}
  {cumsum+=$1; print $2, $1, $1/total * 100 "%", cumsum/total * 100 "%"}' | \
  head -n5 | \
  column -ts,

ls   2470  20.7703%  20.7703%
cd   1509  12.6892%  33.4595%
git  794   6.67676%  40.1362%
vim  468   3.93542%  44.0716%
aws  464   3.90178%  47.9734%
```

The output is the command, number of times types, percentage of all commands typed and cumulative percentage of all commands typed.
So I can see that these 5 are nearly half of all the commands I type!

This only scratches the surface of what's possible; you can filter rows using `grep` (analogous to a `where` clause in SQL) and combine these in novel ways.
For more complex (or robust!) transformations and aggregations it's worthwhile moving to more featureful languages like Python/R/SQL, but being able to quickly generate summaries is a handy tool - especially when something breaks!