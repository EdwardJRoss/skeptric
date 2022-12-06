---
date: '2021-03-11T08:00:00+11:00'
image: /images/docker_resume.png
title: Docker Dependency Managment
---

I have a [personal CV](https://github.com/EdwardJRoss/resume) written in TeX.
I wouldn't use TeX again today, but I've kept it maintained and haven't had reason to migrate it.
Unfortunately the dependencies can be painful; whenever I move to a new machine I need to remember which packages I need to install.
A Dockerfile is a handy way to wrap up the dependencies, and create a container with those dependencies.

Docker is a really handy tool for dependency management.
If you're running on a specific system with a package manager you can use build scripts to set up an environment.
But sometimes there may be conflicts with other packages on your system, or it may be tricky to set up.
A Docker image lets you wrap up all the dependencies in a nice script.

For my TeX setup I already have make commands to produce different formats such as `make pdf`, `make doc`, `make html`, and `make clean` to remove all the files.
To be able to run these entrypoints I use `make` as the entrypoint and pass a `CMD` which defaults to `all`, but can be called with any of the targets.

```
ENTRYPOINT ["make"]
CMD ["all"]
```

I also need to have the files available at runtime; I do this by mounting the current directory as `/home` and setting `WORKDIR /home` in the dockerfile.
Then for example to build the image in a container named `resume` and then build `html` output I would run (from the directory containing the Makefile):

```
docker build -t resume .
docker run -v $(pwd):/home resume html
```

Other than that the Dockerfile just handles the dependencies.
I have some scripts using Pandoc to produce the html and doc outputs, which is only available in Alpine testing edge, which I install like this:

```
RUN apk add --no-cache -X http://dl-cdn.alpinelinux.org/alpine/edge/testing pandoc
```

Here's the whole Dockerfile for this example, which builds a 750MB image:

```
FROM alpine:3.13
RUN apk add --no-cache texlive texlive-dvi ghostscript make
# For HTML and DOCX output
RUN apk add --no-cache -X http://dl-cdn.alpinelinux.org/alpine/edge/testing pandoc
RUN apk add --no-cache python3
WORKDIR /home
ENTRYPOINT ["make"]
CMD ["all"]
```
