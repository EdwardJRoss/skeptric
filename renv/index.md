---
categories:
- R
- blog
date: '2020-11-09T06:58:18+11:00'
draft: true
image: /images/
title: Towards Reproducible R Blogdown With Renv
---

I have a couple of articles that use RMarkdown

https://rstudio.github.io/renv/articles/renv.html

install.packages('renv')

How to build site without serving???

```

Welcome to renv!

It looks like this is your first time using renv. This is a one-time message,
briefly describing some of renv's functionality.

renv maintains a local cache of data on the filesystem, located at:

  - '~/.local/share/renv'

This path can be customized: please see the documentation in `?renv::paths`.

renv will also write to files within the active project folder, including:

  - A folder 'renv' in the project directory, and
  - A lockfile called 'renv.lock' in the project directory.

In particular, projects using renv will normally use a private, per-project
R library, in which new packages will be installed. This project library is
isolated from other R libraries on your system.

In addition, renv will update files within your project directory, including:

  - .gitignore
  - .Rbuildignore
  - .Rprofile

Please read the introduction vignette with `vignette("renv")` for more information.
You can browse the package documentation online at https://rstudio.github.io/renv/.
```


```

* '~/.local/share/renv' has been created.
WARNING: One or more problems were discovered while enumerating dependencies.

static/resources/minhash_s_curves.R
---------------------------------------------------------------------

ERROR 1: static/resources/minhash_s_curves.R:133:1: unexpected '}'
132: 
133: }
     ^
```


```

Installing openssl [1.4.3] ...
	FAILED
Error installing package 'openssl':
===================================

* installing *source* package ‘openssl’ ...
** package ‘openssl’ successfully unpacked and MD5 sums checked
** using staged installation
Using PKG_CFLAGS=
--------------------------- [ANTICONF] --------------------------------
Configuration failed because openssl was not found. Try installing:
 * deb: libssl-dev (Debian, Ubuntu, etc)
 * rpm: openssl-devel (Fedora, CentOS, RHEL)
 * csw: libssl_dev (Solaris)
 * brew: openssl@1.1 (Mac OSX)
If openssl is already installed, check that 'pkg-config' is in your
PATH and PKG_CONFIG_PATH contains a openssl.pc file. If pkg-config
is unavailable you can set INCLUDE_DIR and LIB_DIR manually via:
R CMD INSTALL --configure-vars='INCLUDE_DIR=... LIB_DIR=...'
-------------------------- [ERROR MESSAGE] ---------------------------
tools/version.c:1:10: fatal error: openssl/opensslv.h: No such file or directory
    1 | #include <openssl/opensslv.h>
      |          ^~~~~~~~~~~~~~~~~~~~
compilation terminated.
--------------------------------------------------------------------
ERROR: configuration failed for package ‘openssl’
* removing ‘/home/eross/src/projects/skeptric/renv/library/R-3.6/x86_64-pc-linux-gnu/.renv/1/openssl’
Installing httr [1.4.2] ...
	FAILED
Error installing package 'httr':
================================

ERROR: dependencies ‘curl’, ‘openssl’ are not available for package ‘httr’
* removing ‘/home/eross/src/projects/skeptric/renv/library/R-3.6/x86_64-pc-linux-gnu/.renv/1/httr’
Installing cpp11 [0.2.4] ...
	OK [built from source]
Installing tidyr [1.1.2] ...
	OK [built from source]
Installing rwalkr [0.5.3] ...
	FAILED
Error installing package 'rwalkr':
==================================

ERROR: dependency ‘httr’ is not available for package ‘rwalkr’
* removing ‘/home/eross/src/projects/skeptric/renv/library/R-3.6/x86_64-pc-linux-gnu/.renv/1/rwalkr’
```


```

The following package(s) were not installed successfully:

	[curl]: install of package 'curl' failed
	[openssl]: install of package 'openssl' failed
	[httr]: install of package 'httr' failed
	[rwalkr]: install of package 'rwalkr' failed
```


```
sudo apt install libssl-dev
```


```
install.packages('rwalkr')

Installing curl [4.3] ...
	FAILED
Error installing package 'curl':
================================

* installing *source* package ‘curl’ ...
** package ‘curl’ successfully unpacked and MD5 sums checked
** using staged installation
Package libcurl was not found in the pkg-config search path.
Perhaps you should add the directory containing `libcurl.pc'
to the PKG_CONFIG_PATH environment variable
No package 'libcurl' found
Package libcurl was not found in the pkg-config search path.
Perhaps you should add the directory containing `libcurl.pc'
to the PKG_CONFIG_PATH environment variable
No package 'libcurl' found
Using PKG_CFLAGS=
Using PKG_LIBS=-lcurl
------------------------- ANTICONF ERROR ---------------------------
Configuration failed because libcurl was not found. Try installing:
 * deb: libcurl4-openssl-dev (Debian, Ubuntu, etc)
 * rpm: libcurl-devel (Fedora, CentOS, RHEL)
 * csw: libcurl_dev (Solaris)
If libcurl is already installed, check that 'pkg-config' is in your
PATH and PKG_CONFIG_PATH contains a libcurl.pc file. If pkg-config
is unavailable you can set INCLUDE_DIR and LIB_DIR manually via:
R CMD INSTALL --configure-vars='INCLUDE_DIR=... LIB_DIR=...'
--------------------------------------------------------------------
ERROR: configuration failed for package ‘curl’
* removing ‘/home/eross/src/projects/skeptric/renv/staging/1/curl’
Error: install of package 'curl' failed
Traceback (most recent calls last):
14: install.packages("rwalkr")
13: install(pkgs)
12: renv_install(records, library)
11: renv_install_staged(records, library)
10: renv_install_default(records, library)
 9: handler(package, renv_install_impl(record))
 8: renv_install_impl(record)
 7: withCallingHandlers(renv_install_package_local(record), error = function(e) {
        vwritef("\tFAILED")
        writef(e$output)
    })
 6: renv_install_package_local(record)
 5: renv_install_package_local_impl(package, path, library)
 4: r_cmd_install(package, path, library)
 3: r_exec(package, args, "install")
 2: r_exec_error(package, output, label)
 1: stop(error)
```


```
sudo apt install libcurl4-openssl-dev
```


```
install.packages('rwalkr')
renv::snapshot()
```



```
install.packages('blogdown')
renv::snapshot()
```


```
> renv::snapshot()
* The lockfile is already up to date.
renv::install('blogdown')
```


No Blogdown in renv.lock


Command line:

```
R --no-save -e 'renv::init()'
```