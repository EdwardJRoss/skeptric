---
categories:
- programming
date: '2020-11-20T21:53:10+11:00'
image: /images/non-posix-sh.png
title: Which /bin/sh
---

I tried to run a shell script and got this error:

```
 set: Illegal option -o pipefail
```

I had a quick look and the first line was `#!/bin/sh`, the `-o pipefail` isn't valid across POSIX shells so I would expect that to fail.
More specifically on modern Ubuntu `/bin/sh` is `dash` which doesn't support these `bash` like constructions.

But `/bin/sh` is very different on different systems; on some it is `bash`, on others it's [`ash`](https://en.wikipedia.org/wiki/Almquist_shell) (from which `dash` is derived), and on others it's `ksh` or something else.
The script probably worked on whatever system it was developed on because it had a different `/bin/sh`.
This [has been a problem](https://bugs.launchpad.net/ubuntu/+source/dash/+bug/61463) since Ubuntu switched from `bash` to `dash` in 2006; a lot of scripts assumed they could use `bash` and worked fine, but suddenly stopped working.
People tend to work with an implementation over a specification; if it runs on their machine it's right.

It's easy to fix by changing the shebang, for example to `#/bin/bash`, or the harder way is to make the script POSIX compliant.
The excellent [shellcheck](https://github.com/koalaman/shellcheck) will tell you what parts are not POSIX and makes it easier to chagne.