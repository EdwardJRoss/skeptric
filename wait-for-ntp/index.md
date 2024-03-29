---
categories:
- linux
date: '2020-04-07T20:42:50+10:00'
image: /images/timedatectl.png
title: Waiting for System clock to synchronise
---

When trying to install packages with `apt` on a new Ubuntu AWS EC2 instance I had issues where the signature would fail to verify.
The reason was the system clock was far in the past and so it looked like the signature was signed in the future.
I created a workaround to wait for the system clock to synchronise that solved the problem and could be useful when starting a new machine with time sensitive issues.

```bash
# Wait for ntp to stabilise, so package signatures can be verified
while [[ $(timedatectl status | grep 'System clock synchronized' | grep -Eo '(yes|no)') = no ]]; do
    sleep 2
done
```

This should work on most Linux systems using `systemd` with ntp enabled (e.g. via `timedatectl set-ntp true`).
Then `timedatectl status` could update something like the following:

```
Local time: Tue 2020-04-07 11:02:40 UTC
Universal time: Tue 2020-04-07 11:02:40 UTC
RTC time: Tue 2020-04-07 11:02:41
Time zone: Etc/UTC (UTC, +0000)
System clock synchronized: yes
systemd-timesyncd.service active: yes
RTC in local TZ: no
```

The script above looks for the `System clock synchronized` line and will wait as long as that line has `no` in it.
If it changes to `yes` (or if it can't find that line or the words `yes` or `no` in that line) then the script will coninue.

Putting this at the top of my script before running `apt update` and `apt install` commands I ran as soon as an EC2 instance fixed the signature verification issues I had.