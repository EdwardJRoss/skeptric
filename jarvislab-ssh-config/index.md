---
categories:
- python
date: '2022-11-28T18:37:50+11:00'
image: /images/jarvislabs_api.jpg
title: Automatically updating SSH Config for Jarvislabs
---

[Jarvislabs](https://jarvislabs.ai/) is a very cost efficient cloud GPU provider for deep learning.
One slight issue I ran into is that every time you resume an instance it gets a different SSH host and port.
If you're using their Jupyter notebook interface this is fine, but I wanted to control my own environment more carefully and would SSH into the instance.

I wanted to hide this by hiding it behind an SSH configuration hostname, with the credentials updating automatically when I resume the instance.
Luckily they provide an [API](https://jarvislabs.ai/docs/api/) and so I was able to hack a script to do this in a couple of hours.
This means I can just start the instance from the command line and then run `ssh myinstance` without having to visit the web UI.


We will walk through how we do this, first by getting the instance from the Jarvislabs API, then parsing and updating the SSH config.

# Getting the Jarvislabs Instance

First we need to authenticate to Jarvislabs and retrieve our instance, in order to resume it and get the SSH string.

## Authenticate to Jarvisclient

The jarvisclient library requires a global configuration.
After generating an [API token](https://cloud.jarvislabs.ai/listsshkeys) I set them as environment variables that can be read in (these could be stored in other ways too).

```python
import os
from jlclient import jarvisclient

jarvisclient.token = os.getenv('JARVISLABS_TOKEN')
jarvisclient.user_id = os.getenv('JARVISLABS_USER_ID')

assert jarvisclient.token is not None
assert jarvisclient.user_id is not None
```

## Getting Jarvislabs instance by name

Now we need to get our set up instance to resume and get the SSH details.
The library doesn't provide a way to do get an individual instance, so we search through all the instance and find the one with a matching name.
This particular function assumes that a single instance by that name exists; if it doesn't exist it will return an error.

```python
from jlclient.jarvisclient import Instance, User

def get_instance_by_name(name: str) -> Instance:
    instance = [i for i in User.get_instances() if i.name==name][0]
    return instance
```

Now we have our `Instance` we can `.resume()` it and then update our SSH configuration.

# Updating SSH Config

An instance has an SSH string in the `.ssh_str` attribute (like `ssh -p 1234 root@ssha.jarvislabs.ai`).
We need to extract this and update the SSH configuration file with it.

## Parsing ssh configuration

We can parse this string with some simple regex.
We will wrap it in a simple `SSHConfig` dataclass to make it easy to update the configuration, as explained in the next section.

```python
import re

def parse_ssh_config(s: str) -> SSHConfig:
    match = re.match(r'^ssh -p (?P<Port>[0-9]+) (?P<User>[a-z]+)@(?P<Hostname>[a-z\.]+)$', instance.ssh_str)
    return SSHConfig(**match.groupdict())
```

## Updating SSH Configuration

SSH configuration is documented in the [ssh_config(5) man page](https://linux.die.net/man/5/ssh_config), and is fairly simple to parse.
However it's complicated enough I used the [sshconf](https://github.com/sorend/sshconf) library to update it for me.
It provides a handy `.set(section, **kwargs)` method to update arguments in a given `Host` section of the SSH file.

```python
import dataclasses
from dataclasses import dataclass
from sshconf import SshConfig

@dataclass
class SSHConfig:
    Port: int
    User: str
    Hostname: str

    def update(self, config: SshConfig, section: str) -> None:
        config.set(section, **dataclasses.asdict(self))
```

# Putting it all together

Now we have all the pieces we need to:

1. Get the Jarvislabs instance by name
2. Resume the instance
3. Parse the SSH string from the instance
4. Read the existing SSH configuration
5. Create the SSH config section if it doesn't exist
6. Update the section with the parsed SSH data
7. Save the SSH config file

Here's the script to do it:

```python
from pathlib import Path
from sshconf import read_ssh_config

ssh_config_path = (Path.home() / '.ssh') / 'config'
instance_name = 'myinstance'
ssh_config_section = instance_name

if __name__ == '__main__':
    instance = get_instance_by_name(instance_name)
    instance.resume()

    new_config = parse_ssh_config(instance.ssh_str)
    config = read_ssh_config(ssh_config_path)

    if not config.host(ssh_config_section):
        config.add(ssh_config_section)
    new_config.update(config, ssh_config_section)
    config.save()
```

Now I can easily resume and SSH into Jarvislabs instances from the command line.
It would be nice to expose more of the Jarvislabs API through a command line wrapper and have this as a simple configuration option.