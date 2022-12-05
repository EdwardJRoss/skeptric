---
categories:
- tools
date: '2020-10-15T21:47:40+11:00'
image: /images/keepass.jpg
title: Moving Away From Keepass
---

A password manager is one of the best ways for the majority of people to keep their logins secure.
After using [KeePass](https://keepass.info/) and its derivatives for years, the [Kee Firefox Addon](https://addons.mozilla.org/en-US/firefox/addon/keefox/) dropped support for Keepass and it's now less convenient to use.
After looking at the alternatives I'm going to switch to an online alternative.

One of the most frequent ways people get their accounts hacked is by password reuse.
Their email and password is revealed in some online breach of a website, and then these credentials can be used on other websites.
However it's really hard to remember lots of strong passwords.
A password manager does exactly that; generate and store strong passwords for lots of sites.

A password manager is also a huge target; if you can infiltrate it you can access all their passwords.
For someone who is a very likely target for sophisticated hackers a password manager is a bad idea - it's a liability, and they should take the effort to do something like [diceware](https://theworld.com/~reinhold/diceware.html).
However for the rest of us a password manager is likely worth the cost since it's so convenient you're likely to actually use it.

I've been using KeePass and it's derivatives (such as [KeePassDroid](http://www.keepassdroid.com/) and [KeePassXC](https://keepassxc.org/)) for years.
It's open source, been audited to be relatively secure, and everything is stored in a local file which makes it easy to backup and you know who has control.
However it doesn't have a native way to automatically fill passwords into pages or sync across devices, so I am looking for an alternative.

I was using the Kee Firefox Addon to automatically fill passwords, but in a recent version it's stopped supporting KeePass and become it's own management system that I don't trust or want to use.
Automatic filling is both convenient, but it's also important for security; it prevents you copy pasting into a forged website if you don't carefully check the domain (which can happen with email phishing attacks).

I was using Syncthing to sync the Keepass database between devices, but it seems like Syncthing is unexpectedly opening Tor connections (without being configured to do so) which makes me really suspicious and makes it a no go to use in work devices.
Manually syncing is difficult and error prone, and I have to fall back to typing in passwords from my mobile app which is painfully slow.

So I'm giving up control of open source to find a managed service to handle this for me.
Giving all my passwords to someone else makes me a bit nervous; but I have to rely on the commercial reputation they would lose in a breach to give my confidence that my secrets are relatively secure (for my use case).
Luckily now there's a whole heap of these services from vendor specific solutions like Firefox Sync (which has questionable security regarding encryption) to general password managers like 1Password, LastPass, Bitwarden and Dashlane.
I'm going to look through them and see what seems best for my use case, and start switching to it.