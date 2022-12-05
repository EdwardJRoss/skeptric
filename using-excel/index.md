---
date: 2019-09-25 17:07:45+10:00
image: /images/excel_2013.png
title: A programmer using Excel
---

# Intro

When I was 15 I did a week of work experience with my neighbour, who was an agricultural economist running his own one person business.
I'm still not really sure what an agricultural economist does, but I went out with him to visit his clients to talk through their business, and saw how he analysed their data in his Excel spreadsheet.
It was really closer to an application than a spreadsheet; the interface made it clear where the client was meant to enter their data, it showed some summary output and most of the intermediate calculations were hidden.
This is awful, I thought, a *real* application should be written in a real programming language, like Java.
My neighbour-boss was very diplomatic and gave me a day to prototype a solution.
After a day copying code from the official Java tutorial and the "O'Riley Java Cookbook" I got precisely nowhere and gave up.

15 years later I could comfortably rewrite that application in Java, or a web app in Python or R Shiny.
And I still prefer those solutions for solutions that need to be maintained, or with more complex logic.
But today I sometimes choose to use Excel for client deliverables.
Excel can be a good choice for implementing small modelling and reporting tasks to be used by a domain expert.

# The value of ubiquity

Most of my stakeholders can open, use, and understand any spreadsheet I send them.
They're intimately familiar with the interface, and they can create new calculated fields and graphs to put in presentation packs.
Many of them can follow the logic at a high level and check intermediate steps of the model.
When they can obtain new data then they can update data extracts in the sheet by themselves.

As a developer analyst this means many fewer iterations to get to a usable result, and a much lower setup cost than a custom software solution.
There are no servers to maintain, few user interface concerns and no barrier to entry installing software.
I don't need to iteratively improve the user interface to add the right graphs and summaries, they can do it themselves.
I still need to make sure the model is being understood and my stakeholders are asking the right questions to inform their decisions, but there's a much lower development cost.

# Versatility of Calculations

Excel has a lot of built in functionality that lets you build a variety of calculations.
From Boolean logic, trigonometry and statistical distributions to substrings and date handling there's a lot you can do (although the
[correctness of statistical procedures isn't perfect, it has gotten better](https://link.springer.com/article/10.1007/s00180-014-0482-5)).
With [`VLOOKUP`](https://support.office.com/en-us/article/vlookup-942f678a-1bfc-4ccf-8dfa-f5057ded5c65?ui=en-US&rs=en-US&ad=US) (or better `INDEX` and `MATCH`) you can join data between different tables.
With [pivot tables](https://support.office.com/en-us/article/create-a-pivottable-to-analyze-worksheet-data-a9a84538-bfe9-40a9-a8e9-f99134456576) you can do pretty much anything you could do with a single "Group By" aggregation in SQL (and with a [Data Model](https://support.office.com/en-us/article/Create-a-Data-Model-in-Excel-87E7A54C-87DC-488E-9410-5C75DBCB0F7B) you can do the equivalent of complex joins).

You can also store sets of parameters using [scenario manager](https://support.office.com/en-us/article/switch-between-various-sets-of-values-by-using-scenarios-2068afb1-ecdf-4956-9822-19ec479f55a2), solve equations with [Goal Seek](https://support.office.com/en-us/article/use-goal-seek-to-find-the-result-you-want-by-adjusting-an-input-value-320cb99e-f4a4-417f-b1c3-4f369d6e66c7) or optimise equations with the [Solver add-in](https://support.office.com/en-us/article/define-and-solve-a-problem-by-using-solver-5d1a388f-079d-43ac-a7eb-f63e45925040).

You can even define [custom functions](https://support.office.com/en-us/article/create-custom-functions-in-excel-2f06c10b-3622-40d6-a1b2-b6748ae8231f), [macros](https://support.office.com/en-us/article/automate-tasks-with-the-macro-recorder-974ef220-f716-4e01-b015-3ea70e64937b) or automate things with [VBA](https://docs.microsoft.com/en-us/office/vba/library-reference/concepts/getting-started-with-vba-in-office).
However executing custom code someone has sent you is a huge security risk, and many office viruses have been spread through macros and VBA scripts, so most organisations (wisely) disable them.
The primary benefit of using Excel is it's easy and convenient; if your clients need approval from the security team to access your spreadsheet it's probably better to use a different solution.
They also raise the barrier to entry; unless your function is clearly named and documented it's not going to be clear to most Excel users what it's doing, and it's going to be hard for them to modify it.

![Excel Ribbon showing Macros have been Disabled](/images/macros_disabled.png)



# Clarity of visual calculations

A well laid out workbook can make the flow of calculation visible making it more transparent and easier to see errors.
Most programming environments require an interactive debugger or trawling through debug level logs to see the flow of calculation.
I've often had domain experts notice things in intermediate calculations that showed that the data or the model were wrong, and that weren't obvious from the final result alone.

The logic can be very clear if you use [named formulas](https://support.office.com/en-us/article/define-and-use-names-in-formulas-4d0f13ac-53b7-422e-afd2-abd7ff379c64) so that you can have a formula for Tax like `= Salary * TaxRate` instead of an opaque calculation of Tax as `=C2 * $B$24`.
Using the shortcut CTRL + Backtick (Grave Accent) you can display formulas instead of results to follow the logic of calculations.

However it does take good judgement and discipline to know how to layout a calculation and to use well named fields and ranges.
It's also easy to break a calculation by changing a cell or inserting a row, and when the spreadsheet doesn't all fit on a single screen it becomes hard to spot.
As a model grows it takes up more space over more worksheets, it becomes harder to see how everything fits together, and becomes difficult to understand and maintain.

# Spreadsheet Maintenance

Software that grows needs to be trimmed and pruned to keep it healthy and prevent it from growing out of control.
Excel's low barrier to entry is great for starting out, but undisciplined use makes it really hard to keep working (and updating) correctly.
Spreadsheets tend to be quite fragile, and small changes like adding a row can easily break it in subtle ways.
Tools like [tables](https://support.office.com/en-us/article/overview-of-excel-tables-7ab0bb7d-3a9e-4b56-a3c9-6c94334e492c) can help, but you have to be careful to design robust spreadsheets.

Updating data can be a problem because Excel tries to automatically detect the type of data which can mangle it; [one-fifth of biology papers using Excel contain an error of this sort](https://genomebiology.biomedcentral.com/articles/10.1186/s13059-016-1044-7).
Excel doesn't have a method of version control that lets you compare changes easily, so it's very hard to find out if something has been broken, especially in a spreadsheet too large to fit in a couple of screens.
Once a workbook gets to a sufficient complexity it becomes hard to keep track of what's going on.

# Lack of Abstraction

Abstraction is one of the most powerful tools for building software, and spreadsheets don't really have them.
It's useful to package data transformations in a function/method/subroutine that can be tested and maintained in isolation, and reused in other places.
This allows incrementally building complex software from small pieces that can be more easily verified and changed independently.

Spreadsheets naturally have calculations that build on top of other calculations, making long chains that can't be broken into separate pieces.
If you want to make a change in the middle of the calculation you typically have to then check every step afterwards (and may need to change steps prior to get the data in the right formation).
If you're working in an organisation where custom functions and VBA are allowed then it is possible to build abstractions, and reuse them via [add-ins](https://docs.microsoft.com/en-us/office/dev/add-ins/excel/excel-add-ins-overview). But this doesn't work if you've got security considerations, and you still have to handle versioning and distribution.
There are also add-ins you can buy for advanced analytical procedures like [resampling](http://www.resample.com/download-excel/) and [forecasting](https://www.xlstat.com/en/); but the ecosystem is nowhere near as rich as R or Python, where the functions compose much more easily.

# Limitations

Excel has some [hard limitations](https://support.office.com/en-us/article/excel-specifications-and-limits-1672b34d-7043-467e-8e27-269d656771c3) which can make certain things impossible.
It can take at most 1 million rows and 16 thousand columns per worksheet, which makes it unsuitable for working with larger datasets (although a lot of the time you can get away with sampling).
Spreadsheets don't natively have data structure concepts like HashMaps or Binary Trees and so lookup functions like `VLOOKUP` and `MATCH` run very slowly on bigger worksheets.

While there's a lot of control over the user interface in Excel like [locking](https://support.office.com/en-us/article/Lock-or-unlock-specific-areas-of-a-protected-worksheet-75481b72-db8a-4267-8c43-042a5f2cd93a) or [validation](https://support.office.com/en-us/article/apply-data-validation-to-cells-29fecbcc-d1b9-42c1-9d76-eff3ce5f7249), and you can build simple dashboards with [slicers](https://support.office.com/en-us/article/use-slicers-to-filter-data-249f966b-a9d5-4b0f-b31a-12651785d29d) it always *looks* like Excel (businessy and slightly dated), which might not be what you want.

Frankly you can do a lot of things in Excel (especially with add-ins), but once you get past the core functionality the difficulty rises steeply and returns diminish quickly.
Making systematic changes to code in text is easy; in Excel it's possible (modern versions use XML) but it's really easy to introduce inconsistencies.

# When should you consider Excel

Excel is manageable in the small, but really difficult to manage and maintain in the large.
The benefits of being able to create something quickly that can be used and improved by less technical stakeholders can lead to some quick wins.
But when you need something complex, maintainable, automated or robust it's worth moving to a programming language with good libraries and using testing and version control.
When you just need a pretty reporting tool for simple aggregations that can be maintained by non-programmers then consider a tool like Tableau or PowerBI.

Looking back at my work experience, for an agricultural economist running a sole practice, Excel was the right choice at the time.
He could single-handedly create, update and interpret a functional application that his clients could interact with.
It let him develop and iterate on his product quickly until it was fit for his market, without having to translate his requirements to an external software developer.
He has since turned it into a [Software as a Service](https://p2pagri.com.au/), which would have been much easier since he'd already tested the market.
Looking back I should have been paying more attention to how he was helping his clients grow their businesses rather than getting arrogant with his choice of tools.

If you want to learn how to use Excel more effectively then let Joel Spolsky tell you why ["You Suck at Excel"](https://www.youtube.com/watch?v=0nbkaYsR94c) and check out [how PwC does Excel](https://www.coursera.org/learn/advanced-excel?specialization=pwc-analytics).