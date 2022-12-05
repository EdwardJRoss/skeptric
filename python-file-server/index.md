---
categories:
- web
- python
- javascript
date: '2020-05-19T19:32:41+10:00'
image: /images/cors_error.png
title: Serving Static Assets with Python Simple Server
---

I was trying to load a local file in a HTML page and got a Cross-Origin Request Blocked error in my browser.
The solution was to start a Python web server with `python3 -m http.server`.


I had a JSON file I wanted to load into Javascript in a HTML page.
Looking at StackOverflow I found I found [fetch could do this](https://stackoverflow.com/a/42272155)


```javascript
fetch("test.json")
    .then(response => response.json())
    .then(json => process(json))
```

Where `process` is some function that acts on the data; `console.log` is good for testing.
Reading about [using fetch](https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch) shows that it works similarly to [XMLHttpRequest](https://developer.mozilla.org/en-US/docs/Web/API/XMLHttpRequest), or [jQuery's getJson](https://api.jquery.com/jQuery.getJSON/), with [slightly worse browser coverage](https://caniuse.com/#feat=fetch).
See the [hospodarets article on fetch](https://hospodarets.com/fetch_in_action) for more on using it.

When I tried to do this I opened the HTML file locally in Firefox and in the console got an error message:

```
Cross-Origin Request Blocked: The Same Origin Policy disallows reading the remote resource at file:///...
```

Trying to request a file from HTTP gives this error because it has [security implications](https://developer.mozilla.org/en-US/docs/Web/HTTP/CORS/Errors/CORSRequestNotHttp).

The simplest workaround is to serve "test.json" from a server.
This is trivial with Python's [inbuilt HTTP server](https://docs.python.org/3/library/http.server.html).
Just go to the directory contianing the HTML and Json files in a shell and type `python3 -m http.server` to start a server.
Then you can open the HTML file at `http://localhost:8080/<filename.html>` (or if you call it `index.html` you don't need the extension) and it will be able to load in the JSON file.

In production you would be running this behind a robust HTTP server like Nginx or Apache, but for quickly testing locally Python's inbuilt server is useful.