---
categories:
- jupyter
- python
- r
date: '2020-11-24T20:32:03+11:00'
image: /images/jupyter_favicon.png
title: Setting the Icon in Jupyter Notebooks
---

I often have way too many Jupyter notebook tabs open and I have to distinguish them from the first couple letters of the notebook in front of the Jupyter organge book icon.
What if we could change the icons to visually distinguish different notebooks?

I thought I found a really easy way to set the icon in Jupyter notebooks... but it works in Firefox and not Chrome. 
I'll go through the easy solution works in more browsers and the hard solution.

# More Robust Solution: Changing in Developer Console

This works in at least Chrome and Firefox, but doesn't seem to work with Edge.
If you open the developer console (in Chrome and Firefox press F12 and select console) in your browser you can change the favicon with a little Javascript; to change it to the file `favicon.ico` use:

```js
document.querySelector("link[rel*='icon']").href = "favicon.ico";
```

The site favicon.cc has a list of [top rated favicons](https://www.favicon.cc/?action=icon_list&order_by_rating=1) and it often has base64 encoded versions you can paste straight in without downloading the image, just replace the file with that image string.
For example to get [this palm tree](https://www.favicon.cc/?action=icon&file_id=927844) you can run:

```js
document.querySelector("link[rel*='icon']").href = "data:image/x-icon;base64,AAABAAEAEBAAAAEAIABoBAAAFgAAACgAAAAQAAAAIAAAAAEAIAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAACj7vf+o+73/qPu9/6j7vf+o+73/mWDo/9hoqv+ernC/nq5wv6e1t7+o+73/qPu9/6j7vf+x61d/q2NLf6tjS3+x61d/setXf6j7vf+o+73/op0jP5lg6P/TmqH/2Giq/56ucL+ntbe/p7W3v7HrV3+x61d/q2NLf6tjS3+rY0t/q2NLf6tjS3+rY0t/q2NLf6KdIz+ZYOj/05qh/+tjS3+rY0t/q2NLf6tjS3+rY0t/q2NLf6tjS3+rY0t/q2NLf7bwXL+28Fy/tvBcv7bwXL+inSM/mWDo/9Oaof/28Fy/tvBcv7bwXL+28Fy/tvBcv7bwXL+28Fy/tvBcv7bwXL+////////////////8taA/vLWgP6KdIz+ZYOj/05qh//y1oD+8taA/vLWgP7y1oD+kNRm/vLWgP7y1oD+8taA/vLWgP7/////8taA/vLWgP7y1oD+8taA/mWDo/9lg6P/TmqH//LWgP6Q1Gb+kNRm/ly1ov9ctWv/8taA/vLWgP6C6M/+gujP/vLWgP6Q1Gb+8taA/vLWgP7y1oD+ZYOj/05qh//y1oD+8taA/ly1ov9ctWv/AY8Z/5DUZv5ctWv/sOiC/gGPGf9ctaL/kNRm/oLoz/7y1oD+NFmA/zRZgP9lg6P/NFmA/zRZgP9ctWv/KaY+/wGPGf8ppj7/sOiC/ly1ov8Bjxn/KaY+/1y1a/9ctaL/gujP/jRZgP8bNE//AY8Z/xs0T/80WYD/XLVr/wGPGf8ppj7/sOiC/vLWgP5ctaL/AY8Z/ymmPv+w6IL+sOiC/immPv8ppj7/AY8Z/wGPGf8Bjxn/XLVr/ymmPv8Bjxn/sOiC/ujo6P7o6Oj+8taA/immPv8Bjxn/KaY+/ymmPv8Bjxn/AY8Z/ymmPv8Bjxn/sOiC/gGPGf8Bjxn/KaY+/7Dogv6C6M/+XLWi//LWgP6C6M/+KaY+/wGPGf8Bjxn/KaY+/1y1a/8ppj7/AY8Z/1y1a/+w6IL+/////7Dogv5ctaL/XLVr///////y1oD+XLVr/1y1ov9ctWv/8taA/rDogv5ctWv/KaY+/wGPGf8ppj7/XLWi/4Loz/4ppj7/AY8Z/7Dogv7/////8taA/vLWgP6C6M/+8taA/oLoz/5ctaL/sOiC/ly1a/8ppj7/AY8Z/ymmPv8ppj7/AY8Z/ymmPv9ctaL/gujP/ujo6P7o6Oj+8taA/vLWgP7y1oD+gujP/ly1ov+w6IL+XLVr/ymmPv8Bjxn/XLVr/ymmPv9ctWv/sOiC/ly1ov//////////////////////8taA/vLWgP7y1oD+gujP/rDogv7y1oD+XLVr/7Dogv7y1oD+sOiC/ly1ov/y1oD+AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA==";
```

# Easy way: Doesn't work in all browsers

I found in Firefox that I could change the favicon just by putting this into a cell:

```
%%html
<link rel="icon" type="image/png" href="/path/to/icon.ico">
```

Again the reference can be a base64 encoded image as above.
Another way to run this without magic cells is:

```
from IPython.display import HTML, display
display(HTML('<link rel="icon" type="image/png" href="/path/to/icon.ico">'))
```

For R you have to use the relevant `display_html` function.

```
IRdisplay::display_html(favicon)
```

Unfortunately none of these seem to work in Google Chrome (which Google Analytics tells me there's a 70% chance you're using it right now).
I'm not sure why I can't change the favicon from Jupyter in Chrome; not even executing the javascript in a cell seems to work.
It would be nice to have a cross-browser way to do it within a notebook, because then it could be wrapped in a function that gets a random favicon for easy reuse.