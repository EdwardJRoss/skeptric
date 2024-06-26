---
categories:
- python
date: '2020-03-29T08:00:59+11:00'
image: /images/pdb.png
title: Getting Started Debugging with pdb
---

When there's something unexpected happening in your Python code the first thing you want to do is to get more information about what's going wrong.
While you can use print statements or logging it may take a lot of iterations of rerunning and editing your statements to capture the right information.
You could use a REPL but sometimes it's challenging to capture all the state at the point of execution.
The most powerful tool for this kind of problem is a debugger, and it's really easy to get started with Python's pdb.

I'll cover some basic techniques I use, but check out [the manual](https://docs.python.org/3/library/pdb.html) for all the detail.

# Getting into the debugger

The easiest way to get into the debugger is to invoke your python script with pdb.
So instead of `python mymodule.py` you run `python -m pdb mymodule.py`.
You then will drop straight into a `(pdb)` prompt.
If the script is going to raise an error then just type `c` (for continue), and you'll drop back into the debugger as soon as the issue arises.

If you want to break somewhere an error isn't raised you can set a breakpoint where you want to inspect.
You can do this from within `pdb` with `b` (for [break](https://docs.python.org/3/library/pdb.html#pdbcommand-break)), for example to break at line 92 just type `b 92` or to break when `myfunction` is called type `b myfunction`.
Once you've set your breakpoints you can `c`ontinue until you hit one.

Another way to set a breakpoint is by editing the source file and adding [`breakpoint()`](https://docs.python.org/3/library/functions.html#breakpoint) (for Python versions before 3.7 you will need to use `import pdb; pdb.set_trace()`.
Now when the script is invoked normally (e.g `python mymodeule.py`) it will start a pdb debugger at that line.

You can even use pdb in Jupyter notebooks.
After an error write `%debug` in a new cell and you'll have a debugger.
Just be careful because you can't access stop the debugger from Jupyter, you'll need to exit it (and if you delete the cell you may get into an unrecoverable state).

# Using the debugger

Once you've hit the breakpoint you'll want to have a look around.
The first thing I generally do is run:

```python
locals().keys()
```

This will show the variables available in your local environment.
I use `.keys()` because sometimes a variable will be a giant list that `pdb` spends screens printing out.
If you just want the arguments of the function that you're in type `a` (for args).

You can then run normal Python commands to inspect what is happening, but there are a few caveats.
You can't type multiline commands (sometimes you may want to use a `;` as a statement separator in a function definition).
Many letters are used by the debugger (like `n`, `c`, etc), so if you need to print a variable of this name you can use `p` to print it.
Also lambdas just don't work in `pdb`.
If you need some of these features you can go into a python interpreter by typing `interact`.

# Navigating with the debugger

The debugger lets you know the location of the file you are in and the line.
Often the error will be deep within some called function instead of the code you've been working on.
The debugger let's you navigate around to see what is happening.

The most useful commands are `u` (up) for going higher in the stacktrace.
I will often start by running `u` until I'm in some code I am familiar with.
You can go back down again if you're not sure what's happening with `d`own.

If you ever forget where you are you can type `w` (for where), and it will tell you the file, line and function you are in.
If you want more context for the file you can type `l` to show the source code around the point.
If you need to see more you can type `l <start_line>,<end_line>` to get the source code for those lines.

You can `s`tep down into the current function being called, or you move to the `n`ext line.
When you're finished you can `c`ontinue.
After an error it will restart the program (reloading the source), so you can see the effect of any changes you've made to the source code.

Next time you need to handle an error in your Python code run `python -m pdb` or set a `breakpoint()` and open up the [pdb manual](https://docs.python.org/3/library/pdb.html).
It's a great way to find out what's going on by uncovering your wrong assumptions through interactive querying.