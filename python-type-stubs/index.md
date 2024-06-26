---
categories:
- python
date: '2020-11-26T18:46:49+11:00'
image: /images/archiveiterator_type_stub.png
title: Typechecking with a Python Library That Has No Type Hints
---

Type hints in Python allow statically verifying the code is correct, with tools like [mypy](http://mypy-lang.org/), efficiently eliminating a whole class of bugs.
However sometimes you get the message `found module but no type hints or library stubs`, because that library doesn't have any type information.
It's easy to work around this by adding type stubs.

When you see this error it's worth first checking that there aren't any types already available.
Sometimes a later version of the library may have type annotations, so upgrading may make the message go away.
There are third party type repositories, like [data-science-types](https://pypi.org/project/data-science-types/) which is very helpful if you're using libraries like matplotlib, numpy and pandas.
Other times someone has already written a type stub that's not in a package, like [in the tqdm issue for a type-stub](https://github.com/tqdm/tqdm/issues/260#issuecomment-679214126) (and you can put this in `tqdm.pyi` in the root of your package directory).

If no one else has done the work for you then you have to do it yourself.
Mypy has some [documentation on stub files](https://mypy.readthedocs.io/en/stable/stubs.html#stub-files) and [PEP561 outlines the details of how to do it](https://www.python.org/dev/peps/pep-0561/), but they're a bit heavy to read so I'll work through a simple example.

I was trying to typecheck a library with mypy that used [warcio](https://github.com/webrecorder/warcio).
When I ran mypy I got the following errors:

```
01_fetch_data.py:10: error: Skipping analyzing 'warcio.archiveiterator': found module but no type hints or library stubs
```

At the time of writing [warcio doesn't have type annotations](https://github.com/webrecorder/warcio/issues/117), so we have to write our own stubs.
We can start with creating a blank stub; because it's a module we need to do it in the `warcio` direcory and have a `__init__.pyi` file (otherwise mypy won't find them).

```sh
mkdir warcio
touch warcio/archiveiterator.pyi
touch warcio/__init__.pyi
```

We then get a more specific error about a missing module:

```
01_fetch_data.py:10: error: Module 'warcio.archiveiterator' has no attribute 'ArchiveIterator'
```

We don't need to write stubs for *everything* in the module for it to be typechecked; just the things we use.
To get an idea of what to do it's helpful to look at example stubs that come with mypy in [typeshed](https://github.com/python/typeshed), and check the source code you're stubbing. 
In this case we're using ArchiveIterator and all I was using was initialising it (the `__init__` function) and iterating over it (the `__iter__` and `__next__` function).
So I just took these methods from the [source of warcio.archiveiterator](https://github.com/webrecorder/warcio/blob/master/warcio/archiveiterator.py) and replaced the bodies with `...`:

```python
import six

class ArchiveIterator(six.Iterator):
    def __init__(self, fileobj, no_record_parse=False,
                 verify_http=False, arc2warc=False,
                 ensure_http_headers=False, block_size=BUFF_SIZE,
                 check_digests=False): ...

    def __iter__(self): ...

    def __next__(self): ...
```

Running mypy now gets an error about `Name 'BUFF_SIZE' is not defined`.
In the original code it's imported as `from warcio.utils import BUFF_SIZE`; but if I do that I'll have to stub `warcio.utils` too.
I can just cheat and add a type declaration at the top of the file.

```python
import six

BUFF_SIZE: int

class ArchiveIterator(six.Iterator):
    def __init__(self, fileobj, no_record_parse=False,
                 verify_http=False, arc2warc=False,
                 ensure_http_headers=False, block_size=BUFF_SIZE,
                 check_digests=False): ...

    def __next__(self): ...

    def __iter__(self): ...
```

Surprisingly this is enough to make mypy happy.
However we haven't told it much about the types; so there's not much it can actually check.
For example if I pass the string 'Yes please' to `no_record_parse` mypy will say everything is good.

It's fairly easy to infer the types for `__init__`.
The tricky one is `fileobj`; a quick StackOverflow search [gives the types for filelike objects](https://stackoverflow.com/questions/38569401/type-hint-for-a-file-or-file-like-object).
Normally you pass gzipped files in byte mode to `ArchiveIterator` so it can take at least `typing.BinaryIO`, but I don't know whether it can take text files too and could be `typing.IO`.
Generally it's safer to pick the more restrictive type, and then if we get a type error broaden the type as necessary.
The rest of the types are clear from the default values and the method has no return in the source.

I know from library interface that the iterator yields `ArcWarcRecord`s, and so the remaining methods are easy to annotate types too.
Typing a Generator is [a bit complicated](https://docs.python.org/3/library/typing.html#typing.Generator), it takes 3 types, but when you don't send to the generator or return from it the last two arguments are `None`.
All together this is typed now.

```python
import six
from typing import BinaryIO, Generator

from warcio.recordloader import ArcWarcRecord

BUFF_SIZE: int

class ArchiveIterator(six.Iterator):
    def __init__(self, fileobj:BinaryIO, no_record_parse:bool=False,
                 verify_http:bool=False, arc2warc:bool=False,
                 ensure_http_headers:bool=False, block_size:int=BUFF_SIZE,
                 check_digests:bool=False) -> None: ...

    def __next__(self) -> ArcWarcRecord: ...

    def __iter__(self) -> Generator[ArcWarcRecord, None, None]: ...
```

However now I get a message from mypy that I haven't typed `warcio.recordloader` yet; I need to add annotations for `ArcWarcRecord`.

```
error: Skipping analyzing 'warcio.recordloader': found module but no type hints or library stubs
```

So I can easily create *that* stub in `warcio/recordloader.pyi` and add an `ArcWarcRecord` class to get more specific issues about the methods I use.

```
"ArcWarcRecord" has no attribute "content_stream"
"ArcWarcRecord" has no attribute "rec_headers"
"ArcWarcRecord" has no attribute "rec_headers"
```

In this way you only need to stub the methods you actually use, and the task is manageable.
Trying to annotate the whole warcio codebase is quite a task, although this approach gets you a long way.

At the end with a little work annotating the library you get some notion of type safety.
Of course if you get the types wrong in the stub, particularly the return types, it may pass typecheck but still be wrong.
Another gradually typed language, [Typed Racket](https://docs.racket-lang.org/ts-guide/), installs contracts at the boundaries between typed and untyped code, actually asserting the types are correct.
This could be a way to get assurance you're picking up wrong types early.

To get an idea of how type checking can help in this same codebase I was calculating the end byte given an offset and a length; `end_byte = start_byte + length`.
This was then formatted into a string `f"{start_byte}-{end_byte}"`.
Unfortunately `start_byte` and `length` were strings, so the code ran fine but it *concatenated* the strings rather than adding their numbers.
A type check would have caught this early and clearly pointed out the problem, rather than leaving me scratching my head for a while working out what happened.