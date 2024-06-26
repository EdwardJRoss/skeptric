---
categories:
- linux
date: '2020-03-30T19:47:03+11:00'
image: /images/du.png
title: Disk Usage in Linux with du
---

When your harddrive is filling up the `du` utility is a great way of seeing what's taking up all the space.
It can recursively walk through directories to a maximum depth, and print it in human readable sizes.

I'll normally start by running `df` to see what space is used and available.
It's worth looking at the `Mounted On` column if you don't administer the machine because sometimes there are special partitions for large files.

![Example of df](/images/df.png)

Then I'll navigate to the mounted directory and run:

```sh
du -h -d2 . | sort -hr | head -n20
```

![Example of du](/images/du.png)

The first command `du -h -d2 .` means get the size of all files and directories (`du`) for the current directory `.`, print up them to a maximum depth of 2 (`-d2`) in a human readable format `-h` (e.g. 2.1G instead of 2646 which is the number of 1 MB blocks).
We then `sort` the human readable (`-h`) results in reverse (`-r`) order.
Finally we take the top 20 lines with `head`.
Check out [explainshell](https://explainshell.com) for great breakdowns of commands like [this](https://explainshell.com/explain?cmd=du+-h+-d2+.+%7C+sort+-hr+%7C+head+-n20).

This will then show the 20 largest files/directories up to 2 levels under the current directory which I can then assess for clean up.
This sometimes takes a while to produce any output because it has to go through all the files before `sort` can work.

Be careful if you ever run this in the root directory `/` to exclude things like `/proc` or you'll get a lot of weird errors.

Next time your hard drive is filling up try out `du`; it's available on Linux and Mac and even works in Windows under WSL.