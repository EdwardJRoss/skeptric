---
categories:
- excel
date: '2020-06-14T08:00:00+10:00'
image: /images/excel_progress.png
title: Excel Completion Count
---

I was recently running some simple, but tedious, annotation in Excel.
While it's not a good tool for complex annotation for a problem with a simple textual annotation where you can fit all the information to make a decision in a row it can be effective.
However I needed a way to track progress across the team to make sure we finished on time, and see who needed help.

We had a blank column that was being filled in as the annotation progressed, and each person was working on some set of rows.
To see progress I ended up using a formula like this:

```
=AVERAGE(IF(OFFSET(annotation, 
                   [@[Start Row]]-1,                 0,
                   [@[End Row]] - [@[Start Row]] +1, 1
                   ) = "", 
            0, 1))
```

Where annotation is the first cell of the column being annotated, and Start Row and End Row refer to the row numbers that are to be filled in.

![Example of progress](/images/excel_progress.png)


The way it works is straightforward; suppose Start Row is 2 and End Row is 5

* `annotation` is at the top of the column, say B1
* `OFFSET(annotation, [Start Row] - 1, 0, ...)` gets the range starting at row 1 + (2-1), that is B2
* `OFFSET(annotation, ..., [End Row] - [Start Row] + 1, 1)` gets the rectangle of length (5-2) + 1 = 4 and width 1 (so contains 4 cells, B2, B3, B4 and B5).
* `IF(OFFSET(...) = "", 0, 1)` then turns this into a vector where we get 0 if empty (no annotation) and 1 if filled (some annotation)
* `AVERAGE(IF(...))` then gives the proportion of cells filled, as a percentage between 0% (no annotation) and 100% (all filled)

You can even get cute and put a data bar in that fills as annotation progresses.
This simple trick helped me feel like I was getting somewhere with the annotation I was doing, and helped the team work together to get it completed in time.