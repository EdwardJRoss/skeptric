---
categories:
- programming
date: '2020-11-12T22:01:41+11:00'
image: /images/server_permission.png
title: Run Webserver Without Root
---

You've written your web application or API and you now want to deploy it to a server.
You don't want to run it as root, because if someone finds a vulnerability in the server then it will be trivial for them to take over the system.
However only root has permission to run applications on ports 80 and 443.

There are [a few ways to do this](https://stackoverflow.com/questions/413807/is-there-a-way-for-non-root-processes-to-bind-to-privileged-ports-on-linux), but only a couple that make sense for an interpreted language (like Python, as opposed to a compiled binary).
An easy way is using `authbind` to grant access to the ports.

# Autbind

You've got a service ready to serve on port 80 and/or port 443.
How do we run the application?


```sh
# 1. Install authmind
apt-get install authbind
# 2. Create permission files to set read/write permission
sudo touch /etc/authbind/byport/80
sudo touch /etc/authbind/byport/443
# 3. Change the owner to the user who runs the service
# Here I'm assuming they're called `server`
sudo chown server /etc/authbind/byport/80
sudo chown server /etc/authbind/byport/443

# Run the application
authbind --deep /path/to/app
```

That's all there is to it.
