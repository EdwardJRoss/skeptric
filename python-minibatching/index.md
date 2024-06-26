---
categories:
- python
date: '2020-06-03T07:00:37+10:00'
image: /images/python_minibatch.png
title: Minibatching in Python
---

Sometimes you have a long sequence you want to break into smaller sized chunks.
This is generally because you want to use some downstream process that can only handle so much data at a time.
This is common in stochastic gradient descent in deep learning where you are constrained by the memory on the GPU.
But this is also useful for API calls that can take a list, but can't handle all the data at once.
Or for processing large datasets in batches without exhausting memory, potentially feeding them to other processes.

For example suppose we have the sequence `seq = [1, 2, 3, 4, 5, 6, 7]` and we want to batch it up in size 3.
Then we expect the result `[[1,2,3], [4,5,6], [7]]`.

This is straightforward for a list `seq`:

```python
[seq[i:i+size] for i in range(0, len(seq), size)]
```

However this requires knowing the length of the sequence, which we can't do for a generator.
Then we need a bit more code to track how much of the sequence has been consumed.

```python
def minibatch(seq, size):
    items = []
    for x in seq:
        items.append(x)
        if len(items) >= size:
            yield items
            items = []
    if items:
        yield items
```

Finally you may need to pad the last item so it's the same size as the other batches.
For example `[[1,2,3], [4,5,6], [7, None, None]]`.
While it would be easy to update the code above to handle this in the last yield, conceptually this is a separate function.

```python
def pad(items, value, length):
    return items + [value] * (length - len(items))
```

So for example `pad([7], None, 3)` is `[7, None, None]`.
Then you could modify the minibatch function to use the `pad` function in the final `yield`.
Another approach would be to do it with function composition:

```python
def minibatch_pad(seq, size, pad_value):
    return map(lambda x: pad(x, pad_value, size),
               minibatch(seq, size))
```

This is actually a really bad idea in Python.
The biggest reason is if there is a problem it's very hard to follow the stack trace.
While this kind of solution would be common in a lisp, it's quite uncommon in Python for this reason.
Less importantly it's a little bit less computationally efficient.

For completeness here is the padding solution.

```python
def minibatch_pad(seq, size, pad_value):
    items = []
    for x in seq:
        items.append(x)
        if len(items) >= size:
            yield items
            items = []
    if items:
        yield pad(items, size, pad_value)
```