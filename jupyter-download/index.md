---
categories:
- python
- jupyter
date: '2020-09-11T08:00:00+10:00'
image: /images/jupyter_download_button.png
title: Downloading files from Jupyter Notebook
---

You've done an analysis and generated an output file in a Jupyter notebook.
How do you get it down to your computer?
For a local server you could find it in your filesystem, or for a remote server copy it with something like scp.
But there are easier ways.

You can download individual *files* from the file navigator (which you can get to by clicking on the Jupyter icon in the top left corner).
You just need to click the checkbox next to the file you want to download and then click the "download" button at the top of the pane.
The only drawback is it only works for individual files; not directories or multiple files.

![Download interface in Jupter](/images/jupyter_download_select.png)

An even easier way in a Python notebook is to use FileLink:

```python
from IPython.display import FileLink
FileLink(filename)
```

then a link will display below the cell.

If you want links to multiple files in a single cell you need to use `display` to show them all (otherwise it only gets shown if it's the last line of the cell):

```python
from IPython.display import FileLink
display(FileLink(filename1))
display(FileLink(filename2))
```

What if you want to download multiple files or a whole directory?
You don't really want to have to click "download" a whole bunch of times.
But you can easily put them all in a zipfile in Python, and then download the single zipfile as before.
For example if you wanted to download all the csvs in the current directory:

```python
from zipfile import ZipFile
from pathlib import Path
zipname = 'sample.zip'
filenames = Path.glob('*.csv')
with ZipFile(zipname, 'w') as zipf:
    for name in filenames:
        zipf.write(name)
```

If you want to download a whole directory there's an even easier way with `shutil.make_archive`.

```python
from shutil import make_archive
make_archive('sample.zip', 'zip', directory)
```

Then you can download the files very quickly via FileLink without ever leaving the Jupyter notebook.