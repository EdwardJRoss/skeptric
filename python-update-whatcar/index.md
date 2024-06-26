---
categories:
- whatcar
- python
- programming
date: '2020-11-05T10:49:59+11:00'
image: /images/torchvision_nightly_error.png
title: 'Updating a Python Project: Whatcar'
---

The hardest part of programming isn't learning the language itself, it's getting familiar with the gotchas of the ecosystem.
I recently updated my [whatcar car classifier](http://www.whatcar.xyz) in Python after leaving it for a year and hit a few roadblocks along the way.
Because I'm familiar with Python I knew enough heuristics to work through them quickly, but it takes experience with running into problems to get there.

I thought I had done a good job of making it reproducible by creating a Dockerfile for it.
However I hadn't stored the built container anywhere, and when I tried to rebuild it the build failed, complaining something about pip can't find `torchvision-nightly`.
Whatcar uses fastai v1 for the classifier, I had version locked it to `1.0.18` because I knew it was changing in unstable ways.
At the time torchvision was new and it was relying on nightly builds, which probably don't exist anymore now that it is stable.
It seemed the easiest way forward was to upgrade to the most recent version of fastai and Pytorch.

Updating to the latest version of fastai was straightforward, but getting a CPU build of Pytorch is a bit trickier.
Because I'm just serving the Pytorch model there's little advantage of running on a GPU because it will only be scoring one image at a time; it can't batch them up to evaluate in parallel on the GPU.
For [technical reasons](https://github.com/pytorch/pytorch/issues/26340) the CPU build of Pytorch isn't on PyPI.
However it is hosted by Pytorch, and [as discussed in a Pytorch issue](https://github.com/pytorch/pytorch/issues/29745) it can be added to the `requirements.txt` by adding `--find-links`.
Interestingly [you can't specify pip to use an external source in `setup.py`](https://stackoverflow.com/questions/57689387/equivalent-for-find-links-in-setup-py) without passing a flag, so you would need a custom install script.
Anyway I stuck with `requirements.txt` and updated it to look like the following.

```
--find-links https://download.pytorch.org/whl/torch_stable.html
torch==1.7.0+cpu
torchvision==0.8.1+cpu
fastai==1.0.61
```

Finally I could install the dependencies and test them.
I had stored the 90MB model in Google Cloud Storage, but have since removed the account, so I just added it to the repository because it's just [below Github's 100MB limit](https://docs.github.com/en/free-pro-team@latest/github/managing-large-files/conditions-for-large-files) (Gitlab.com has [more generous limits](https://docs.gitlab.com/ee/user/gitlab_com/index.html#account-and-limit-settings)).
While a lot of people will recommend using [LFS](https://git-lfs.github.com/) for files this size, in my experience it's really easy to get in confusing situations with LFS; the files come down as a hash before you initialise LFS (which is easy to forget to do, or do wrong especially when doing many git operations) and it can take some time to work out what's going wrong.
LFS (or other external storage) would matter with lots of changing large files; it makes it easy to checkout just the most recent version than the whole history, which can save on bandwidth, but for this case it's not worth the hassle.

Now when I tried to run it I got an error about `tfms` getting multiple arguments.
A quick google of the error message revealed a [fast.ai forum post](https://forums.fast.ai/t/transform-got-multiple-values-for-argument-tfms-lesson-2/37975/3A) which worked around the problem by changing `tfms` to `ds_tfms` (this kind of thing is why I version locked fastai in the first place).
After that it worked locally.

All up it took me about half an hour to get it working locally, but only because I could make educated guesses about why a lot of the errors were occurring.
But I still had to update it on the server, which surprisingly took me just as long.

Even though I had a Docker container I was running on the server via a virtualenv.
I'm not entirely comfortable about securely deploying with Docker, so I didn't.
I uploaded my new files and tried to upgrade the server.

```
pip install --upgrade requirements.txt
```

However I got some long and scary error message about Bottleneck failing to install.
So I created a fresh virtual environment (`python3 -m venv ~/.venvs/deploy2`) and tried again.
This time I got an error about being out of space, despite having plenty of free space on the server.
Undaunted I deleted empty virtualenvs and `~/.cache` which contains cached versions of downloaded packages from pip (among other things).
Again I got an error about Bottleneck failing to install which I couldn't make much of; but I noticed the message about pip being out of date.
So I upgraded pip with `pip install --upgrade pip` and tried installing the requirements again and it worked like magic.
I reloaded the server and everything was working as expected.

My lesson from all this is when you work with a system a lot you learn how things work by hitting edge cases, and learn heuristics for solving them.
Knowing where pip's cache was allowed me to remove it, knowing a bit about fastai allowed me to quickly navigate the problems that came up in it.
I find this frequently, the first time I had an encoding error (e.g. something not in valid UTF-8 in the middle of a large file) it took me days to work out; now I can generally sense it from the error message and resolve it quickly.
This is often the hardest thing about moving to a new ecosystem; when learning some Haskell I spent days dealing with Stack errors.
I'm sure if I ran a Haskell server in production I'd hit a thunk leak someday which would be incredibly hard to debug the first time.
These things are surmountable, but it's the cost of working with new systems.