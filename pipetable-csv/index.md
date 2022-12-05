---
categories:
- emacs
- data
date: '2020-04-29T21:30:05+10:00'
image: /images/pipetable.png
title: Pipetable to CSV
---

Sometimes I get out pipe tables in Emacs that I want to convert into a CSVto put somewhere else.
This is really easy with regular expressions.

I often get data output from an SQL query like this


```
 text         | num  | value
--------------+------+-------------
   Some text  |  0.3 | 0.2
   Rah rah    |  7   | 0.00123(2 rows)
```

Running `sed 's/\(^  *\| *|\|(.*\) */,/g'` gives:

```
,text,num,value
--------------+------+-------------
,Some text,0.3,0.2
,Rah rah,7,0.00123,
```

I can delete the divider and then use as a CSV.
Even better I can run this same regular expression in Vim or Emacs Evil mode as an Ex command.
This won't work if the data contains parentheses or pipes, but is useful for quick extracts.