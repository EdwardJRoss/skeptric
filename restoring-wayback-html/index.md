---
categories:
- webrefine
date: '2021-12-01T11:02:27+11:00'
image: /images/wayback_html.png
title: Restoring Wayback Machine HTML
---

The [Internet Archive's Wayback Machine](https://web.archive.org/) is a digital archive of a large portion of the internet (hundres of billions of web pages).
However they don't store the webpage in its original form, but make some changes to the page to make it easier to view as it was at the time; for example replacing links to images, CSS and Javascript with their archived versions.
But how exactly do they change the HTML, and how do we get the original version?

I originally came across this when fetching resources from the Internet Archive through its [CDX Server](https://github.com/internetarchive/wayback/tree/master/wayback-cdx-server).
The server response includes a SHA-1 digest, but when I tried to recalculate it on the content I got a different value.
When I searched for why I came [across an Internet Archive post](https://archive.org/post/1009990/cdx-digest-not-accurately-capturing-duplicates) explaining the digest has the SHA-1 of the original content, not what's in the Wayback Machine.

> As you may have guessed, downloading all instances of a webpage, and hashing them yourself, would be worse than relying on the CDX digest. That is because all the instances of the webpage are guaranteed to be different, because the Wayback Machine replaces all links by internal hyperlinks. These urls contain timestamps, and the timestamps obviously differ.

However it turns out to be [trivial](https://stackoverflow.com/a/45045834) to get the original content; if the Wayback version is at `http://web.archive.org/web/<timestamp>/<url>` then the original capture is at `http://web.archive.org/web/<timestamp>id_/<url>`.

> The Internet Archive allows us to retrieve the raw version of web pages. For example, if you have this URL (https://web.archive.org/web/20170204063743/http://john.smith@example.org/), replace the timestamp 20170204063743 with 20170204063743id_ (so the modified URL will look like https://web.archive.org/web/20170204063743id_/http://john.smith@example.org/) then you will get the original HTML without any additional comments added by the Internet Archive.

But I only learned that after spending time trying to reverse engineer the Wayback HTML, and the rest of the article covers what the changes are.

# About a test case

To work out what was happening I needed a small page and so I used my [about page](http://web.archive.org/web/20211120235913/https://skeptric.com/about/) page.

Searching the Internet Archive CDX I get a recent capture:

```python
import requests
r = requests.get('http://web.archive.org/cdx/search/cdx',
                 params={'url': 'skeptric.com/about/', 'output': 'json'})
captures = r.json()

import pandas as pd
df = pd.DataFrame(captures[1:], columns=captures[0])
```

This gives a capture of the page from 2020-11-12:

|    | urlkey              |      timestamp | original                    | mimetype   |   statuscode | digest                           |   length |
|---:|:--------------------|---------------:|:----------------------------|:-----------|-------------:|:---------------------------------|---------:|
|  0 | com,skeptric)/about | 20211120235913 | https://skeptric.com/about/ | text/html  |          200 | Z5NRUTRW3XTKZDCJFDKGPJ5BWIBNQCG7 |     3266 |

We can check the base 32 encoded SHA-1 digest against a current snapshot:

```python
from hashlib import sha1
from base64 import b32encode

def sha1_digest(content: bytes) -> str:
    return b32encode(sha1(content).digest()).decode('ascii')

original_url = f'http://web.archive.org/web/{record.timestamp}id_/{record.original}'
original_content = requests.get(original_url).content
sha1_digest(original_content)
```

This gives `Z5NRUTRW3XTKZDCJFDKGPJ5BWIBNQCG7` which matches the record.

Now we can get the Wayback Machine version of the content by inserting the timestamp and original URL

```python
record = df.iloc[0]
wayback_url = f'http://web.archive.org/web/{record.timestamp}/{record.original}'
wayback_content = requests.get(wayback_url).content

sha1_digest(wayback_content)
```

This gives us a different digest: `DEXQJ2HFM7EYGOWJ6W6FPKIJC4V3VXEE`.

# Headers and footers

Looking at the start of `wayback_content` there's a bunch of Internet Archive Javascript and CSS:

```html
<!DOCTYPE html>
<html lang="en-us">
<head><script src="//archive.org/includes/analytics.js?v=cf34f82" type="text/javascript"></script>
<script type="text/javascript">window.addEventListener('DOMContentLoaded',function(){var v=archive_analytics.values;v.service='wb';v.server_name='wwwb-app213.us.archive.org';v.server_ms=279;archive_analytics.send_pageview({});});</script>
<script type="text/javascript" src="/_static/js/bundle-playback.js?v=UfTkgsKx" charset="utf-8"></script>
<script type="text/javascript" src="/_static/js/wombat.js?v=UHAOicsW" charset="utf-8"></script>
<script type="text/javascript">
  __wm.init("http://web.archive.org/web");
  __wm.wombat("https://skeptric.com/about/","20211120235913","http://web.archive.org/","web","/_static/",
	      "1637452753");
</script>
<link rel="stylesheet" type="text/css" href="/_static/css/banner-styles.css?v=omkqRugM" />
<link rel="stylesheet" type="text/css" href="/_static/css/iconochive.css?v=qtvMKcIJ" />
<!-- End Wayback Rewrite JS Include -->

    <meta charset="utf-8"/>
    <meta http-equiv="X-UA-Compatible" content="IE=edge"/>



    <title>About Skeptric · </title>

    <meta name="HandheldFriendly" content="True"/>
    <meta name="viewport" content="width=device-width, initial-scale=1.0"/>


    <link rel="stylesheet" href="http://web.archive.org/web/20211120235913cs_/https://skeptric.com/style.main.min.5ea2f07be7e07e221a7112a3095b89d049b96c48b831f16f1015bf2d95d914e5.css"/>
```

To get our original content you'd have to strip everything from the first `<script` tag through to the helpful `End Wayback Rewrite JS Include`.
Here's a very rough script to do it:

```python
def remove_wayback_header(content):
    _start = b'<script src="//archive.org/includes/analytics.js'
    _end = b'<!-- End Wayback Rewrite JS Include -->\n'
    start_idx = content.find(_start)
    end_idx = content.find(_end)
    if start_idx < 0 or end_idx < 0:
        raise ValueError("Could not find")
    return content[:start_idx] + content[end_idx+len(_end):]
```

Similarly if you look at the end there's more boilerplate about the archival:

```html
</footer>

    </div>

</body>
</html>
<!--
     FILE ARCHIVED ON 23:59:13 Nov 20, 2021 AND RETRIEVED FROM THE
     INTERNET ARCHIVE ON 00:41:42 Dec 01, 2021.
     JAVASCRIPT APPENDED BY WAYBACK MACHINE, COPYRIGHT INTERNET ARCHIVE.

     ALL OTHER CONTENT MAY ALSO BE PROTECTED BY COPYRIGHT (17 U.S.C.
     SECTION 108(a)(3)).
-->
<!--
playback timings (ms):
  captures_list: 198.782
  exclusion.robots: 0.079
  exclusion.robots.policy: 0.072
  RedisCDXSource: 2.673
  esindex: 0.007
  LoadShardBlock: 177.421 (3)
  PetaboxLoader3.datanode: 81.052 (4)
  CDXLines.iter: 16.33 (3)
  load_resource: 76.041
  PetaboxLoader3.resolve: 25.907
-->
```

We can similarly strip out everything from the file archival comment:

```python
def remove_wayback_footer(content):
    _prefix = b'</html>\n'
    _start = _prefix + b'<!--\n     FILE ARCHIVED ON '
    start_idx = content.find(_start)
    if start_idx < 0:
        raise ValueError("Could not find")
    return content[:start_idx + len(_prefix)]
```

# Restoring Links

The links in the Wayback Machine versino of the webpage are prefixed with http://web.archive.org/web/<TIMESTAMP>, with an extra cs_ for CSS, and js_ for Javascript, and im_ for images.
For example some of the links can be found with:

```python
re.findall(b'(?:href|src)="([^"]*)"', wayback_content)
```

This gives results including:

```
http://web.archive.org/web/20211120235913cs_/https://skeptric.com/style.main.min.5ea2f07be7e07e221a7112a3095b89d049b96c48b831f16f1015bf2d95d914e5.css
http://web.archive.org/web/20211120235913/https://skeptric.com/
/web/20211120235913/https://skeptric.com/about/
/web/20211120235913/https://skeptric.com/
http://web.archive.org/web/20211120235913/https://www.whatcar.xyz/
http://web.archive.org/web/20211120235913js_/https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js
```

So we can remove the prefixes:

```python
def remove_wayback_links(content: bytes, timestamp: str) -> bytes:
    # Remove web links
    timestamp = timestamp.encode('ascii')
    content = content.replace(b'http://web.archive.org', b'')
    for prefix in [b'', b'im_', b'js_', b'cs_']:
        content = content.replace(b'/web/' + timestamp + prefix + b'/', b'')
    return content
```

# And the rest

```python
def remove_wayback_changes(content, timestamp):
    content = remove_wayback_header(content)
    content = remove_wayback_footer(content)
    content = remove_wayback_links(content, timestamp)
    return content
```

We can then compare the cleaned wayback content with the original using `seqmatcher` (see [side-by-side diffs in Jypyter](/python-diffs) for a fancier solution).
For every area where the two are different we print the original and then the cleaned wayback version, with an additional 20 tokens of context on either side:

```python
from difflib import SequenceMatcher
seqmatcher = SequenceMatcher(isjunk=None,
                             a=original_content,
                             b=clean_wayback_content,
                             autojunk=False)

context_before = context_after = 20

for tag, a0, a1, b0, b1 in seqmatcher.get_opcodes():
        if tag == 'equal':
            continue

        a_min = max(a0 - context_before, 0)
        a_max = min(a1 + context_after, len(seqmatcher.a))
        print(seqmatcher.a[a_min:a_max])

        b_min = max(b0 - context_before, 0)
        b_max = min(b1 + context_after, len(seqmatcher.b))
        print(seqmatcher.b[b_min:b_max])
        print()
```

This yields a set of very small changes; here they are:

* Removed trailing whitespace in tags
* Made relative links absolute
* Added a trailing / to the domain URL

```
meta charset="utf-8" />\n    <meta http-eq
meta charset="utf-8"/>\n    <meta http-eq

e" content="IE=edge" />\n\n    \n    \n    <t
e" content="IE=edge"/>\n\n    \n    \n    <t

ndly" content="True" />\n    <meta name="v
ndly" content="True"/>\n    <meta name="v

, initial-scale=1.0" />\n\n    \n    <link r
, initial-scale=1.0"/>\n\n    \n    <link r

015bf2d95d914e5.css" />\n<script async src
015bf2d95d914e5.css"/>\n<script async src

"menuitem"><a href="/about/">About</a></
"menuitem"><a href="https://skeptric.com/about/">About</a></

"menuitem"><a href="/">Home</a></li>\n
"menuitem"><a href="https://skeptric.com/">Home</a></li>\n

https://skeptric.com">skeptric.com</a>.<
https://skeptric.com/">skeptric.com</a>.<
```

What's interesting about this is there's no way to recover this information without the original; there's no way of knowing for sure where the trailing whitespace is (you could *search* for it by matching against the SHA-1, but it would be expensive).
It's good that the Internet Archive provide an original version of the HTML as well!

For this case I wrote a little script that would munge the original content into something closer to what the Wayback Machine emits, but it wouldn't be robust enough to work for other captures:

```python
import re
def wayback_normalise_content(content, base_url):
    url = base_url.encode('ascii')
    content = re.sub(b' */>', b'/>', content)
    content = content.replace(b'href="/', b'href="' + url + b'/')
    content = re.sub(b'href="' + url + b'"', b'href="' + url + b'/"', content)
    return content

assert wayback_normalise_content(original_content, 'https://skeptric.com') == clean_wayback_content
```

If you want to try this at home there's a [Jupyter Notebook](https://nbviewer.org/github/EdwardJRoss/skeptric/blob/master/static/notebooks/restoring_wayback_html.ipynb) (or you can [view it in your browser](/notebooks/restoring_wayback_html.html)).