---
categories:
- python
date: '2021-04-12T08:00:00+10:00'
image: /images/imap_tools.png
title: Reading Email in Python with imap-tools
---

You can use Python to read, process and manage your emails.
While most email providers provide autoreplies and filter rules, you can do so much more with Python.
You could download all your PDF bills from your electricity provider, you could parse structured data from emails (using e.g. BeautifulSoup), sort or filter by sentiment, or even do your own personal analytics [like Steven Wolfram](https://writings.stephenwolfram.com/2012/03/the-personal-analytics-of-my-life/).

The easiest tool I've found for reading emails in Python is [imap_tools](https://github.com/ikvk/imap_tools).
It has an elegant interface to communicate with your email provider using IMAP (which almost every email provider will have).

First you access the MailBox; for which you need to get the imap server and login credentials (username and password).
You should be able to find this in your email providers help or settings (e.g. [here's a guide for Gmail](https://support.google.com/a/answer/9003945)).


```python
from imap_tools import MailBox, AND

# Server is the address of the imap server
mb = MailBox(server).login(user, password)
```

Then you can search for messages based on [RFC 3501 Search Criteria](https://tools.ietf.org/html/rfc3501#section-6.4.4).
There are lots of examples in the [imap_tools README](https://github.com/ikvk/imap_tools#search-criteria); you can search based on the sender, subject, text, date and others.


```python
# Fetch all unseen emails containing "electricity.com" in the from field
# Don't mark them as seen
# Set bulk=True to read them all into memory in one fetch
# (as opposed to in streaming which is slower but uses less memory)
messages = mb.fetch(criteria=AND(seen=False, from_="electricity.com"),
                        mark_seen=False,
                        bulk=True)
```

Then you can access things like the subject, from address, date, and text and HTML content using [simple attributes](https://github.com/ikvk/imap_tools#email-attributes).

```python
files = []
for msg in messages:
    # Print form and subject
    print(msg.from_, ': ', msg.subject)
    # Print the plain text (if there is one)
    print(msg.text)
    # Add attachments
    files += [att.payload for att in msg.attachments if att.filename.endswith('.pdf')]
```

It also handles [actions](https://github.com/ikvk/imap_tools#actions-with-emails) on emails such as flagging as seen, moving and deleting messages.

## Alternatives

Python has the built in [imaplib](https://docs.python.org/3/library/imaplib.html) for IMAP and [email](https://docs.python.org/3/library/email.html) for processing emails.
Unfortunately they're quite low level and require a bit more work to use than imap_tools.

```python
import imaplib
import email

mb = imaplib.IMAP4_SSL(server)
rv, mesasge = mb.login(user, password)
# 'OK', [b'LOGIN completed']
rv, num_emails = M.select('Inbox')
# 'OK', [b'22']

# Get unread messages
rv, messages = M.search(None, 'UNSEEN')
# 'OK', [b'21 22']

# Download a message
typ, data = M.fetch(b'21', '(RFC822)')

# Parse the email
msg = email.message_from_bytes(data[0][1])
print(msg['From'], ":", msg['Subject'])

# Print the Plain Text (is this always the plain text?)
print(msg.get_payload()[0].get_payload())
```

## Dealing directly with Mailfiles

Another alternative would be to download all the messages to your filesystem and directly manipulate the files.
You could run your own [Postfix Server](http://www.postfix.org/) to receive mail, or use [isync/mbsync](http://isync.sourceforge.net/), or it's slower cousin [offlineimap](http://www.offlineimap.org/) to sync the emails to files like I outline in [reading email in Emacs](/emacs-email).

These are a bit harder; for Postfix you've got to make sure your server is up or you'll lose emails.
For mbsync and offlineimap there's a bit of complex configuration, and if you do it wrong you can do mess up your emails on the server.
But if you want to do big batch processing on all your emails this may be an alternative to consider.
