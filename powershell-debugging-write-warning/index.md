---
categories:
- programming
date: '2020-04-23T12:23:01+10:00'
image: /images/write-warning.png
title: Powershell Debugging with Write-Warning
---

I had to debug some Powershell, without knowing anything about it.
I found `Write-Warning` was the right tool for printline debugging.
This was enough to resolve my issue.

I first tried `Write-Output` but apparently it [doesn't work inside a function](https://community.spiceworks.com/topic/2189940-powershell-write-output-in-function-with-return-value) which I found misleading for a while (at first I thought that it wasn't getting to the function).
`Write-Warning` worked straight away and I could see in bright yellow what was going on.

I traced the call with `Write-Warning "At this point"` statements and looked into variables with `Write-Warning "$Variable"` statements.

One interesting thing with Powershell is they can have [`CmdletBinding`](https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.core/about/about_functions_cmdletbindingattribute) at the start of the function.
Everything in to the end of the binding should be considered part of the function and you should only put your debugging statements beneath it (so starting at the `...` below):

```
function MyFunction
{
    [CmdletBinding()]
    param (
    [Parameter()][Type] $Parameter
    )
    
    ...
```

In the end I found the issue was some JSON data was being string interpolated without being properly escaped.
A quick search found [`ConvertTo-Json`](https://docs.microsoft.com/en-us/powershell/module/microsoft.powershell.utility/convertto-json) resolved the issue.

It's sometimes amazing how far you can get with print statement debugging knowing generically how programming languages work (though it may be much trickier in a non strict language like Haskell!).