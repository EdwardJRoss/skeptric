---
categories:
- python
date: '2021-07-04T08:00:00+10:00'
image: /images/typer_order.png
title: Setting the Order of Commands in Typer
---

[Typer](https://typer.tiangolo.com/) is a nice application for succinctly building Python CLIs built on top of [Click](https://click.palletsprojects.com/).
However when you've got subcommands they're listed in alphabetical order.
It would be nice to have the commands ordered in the same way they are in the code, so you can present them to the user in the clearest order to read.

There's a simple solution for this in Python 3.6+ (for older versions outside of C Python you may need a [more elaborate version](https://github.com/pallets/click/issues/513#issuecomment-301046782) with OrderedDict).

```python
import typer
import click

class NaturalOrderGroup(click.Group):
    def list_commands(self, ctx):
        return self.commands.keys()

app = typer.Typer(cls=NaturalOrderGroup)
```

That's all there is to it, with that small change the commands will appear in the same order they are written in the code.