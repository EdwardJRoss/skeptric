---
categories:
- python
date: '2020-07-02T08:00:00+10:00'
image: /images/atomic_file_writer.png
title: Only write file on success
---

When writing data pipelines it can be useful to [cache intermediate results](/caching-pipelines) to recover more quickly from failures.
However if a corrupt or incomplete file was written then you could end up caching that broken file.
The solution is simple; only write the file on success.

A strategy for this is to write to some temporary file, and then move the temporary file on completion.
I've wrapped this in a Python context manager called `AtomicFileWriter` which can be used in a `with` statement in place of `open`:

```python
with AtomicFileWriter(dest_name) as output:
    output.write(...)
```

In the implementation we create a temporary file in the same location by appending `.tmp` to the filename.
If the context is successfully exited this file is closed and moved to the desired destination.
Otherwise if there is an error then the filehandle is closed and the temporary file is deleted.


```python
import os
class AtomicFileWriter:
    """Writes a file to filename only on successful completion"""
    def __init__(self, filename):
        self.filename = filename
        self.temp_filename = str(filename) + '.tmp'

    def __enter__(self):
        self.filehandle = open(self.temp_filename, 'x')
        return self.filehandle

    def __exit__(self, exc_type, exc_value, exc_traceback):
        if exc_type is None:
            self.filehandle.close()
            os.replace(self.temp_filename, self.filename)
        else:
            try:
                self.filehandle.close()
            finally:
                os.unlink(self.temp_filename)
```


This should be safe; moving a file with `os.replace` should be atomic.
It's pretty unlikely there will already be a file with `.tmp` at the end.
There are likely some conditions under which the temporary file won't be cleaned up (e.g. under a `kill -9`).
Looking at [the tempfile source code](https://github.com/python/cpython/blob/master/Lib/tempfile.py) it looks like there's more edge cases I'm likely missing (on some systems), but it works well enough.

I decided to not use [tempfile](https://docs.python.org/3/library/tempfile.html) because:

* The temporary directory may be on a different partition increasing risks of failure (out of disk space, issues moving the file accross an NFS)
* Having the file with a predictable name in the same directory makes it easier to monitor the progress

But using a predictable name has increased the risks of filename collision.
Using the 'x' mode to [open](https://docs.python.org/3/library/functions.html#open) the file means it will fail if the file already exists.
This stops a kind of failure where two processes try to write to the file at the same time leading to corruption.
It does mean that if a `.tmp` file doesn't get deleted on exit it has to be manually cleaned up.


I've been successfully using this with the pattern to only write if the file doesn't exist:

```
from pathlib import Path
for dest_path in Path(dest_dir).glob('*')
    if dest_path.exists():
        continue
    with AtomicFileWriter(dest_path) as output:
        output.write(...)
```