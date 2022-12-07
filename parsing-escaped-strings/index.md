---
categories:
- python
date: '2020-06-16T08:00:00+10:00'
image: /images/DFAexample.svg
title: Parsing Escaped Strings
---

Sometimes you may have to parse a string with backslash escapes; for example `"this is a \"string\""`.
This is quite straightforward to parse with a state machine.

The idea of a state machine is that the action we need to take will change depending on what we have already consumed.
This can be used for [proper regular expressions](/regular-expressions-automata-monoids) (without special things like lookahead), and the ANTLR4 parser generator can maintain a stack of "modes" that can be used similarly.

For escapes this is simple, we're either in an escape or we're not.
If we're in an escape we just consume the next character (if we were interpreting the string we might need to actually transform the character).
If we're not in an escape then we look out for an escape or a closing quote.
In an imperative (Python) implementation we can keep track of the state with a variable:

```python
def find_unescaped_char(text, idx=0, char='"', escapechar='\\'):
    escaped = False
    for offset, c in enumerate(text[idx:]):
        if escaped:
            escaped = False
        elif c == escapechar:
            escaped = True
        elif c == char:
            return offset + idx
```

For example `find_unescaped_char(r'\""')` returns 2 (since the first unescaped quote is at character 2), but `find_unescaped_char(r'\\"")` also returns 2 because it's the backslash that's escaped.

There are lots of different ways to implement this; for example you could treat each state as a separate object that then passes the remainder of the string to the next object, which has it's own parsing function.
Another option is as functions that call each other, or return the remainder of the string and the next function to parse.
These solutions make it much simpler to manage lots of different states because you keep the parsing logic for each state in a separate object or function.

Another example is trying to parse a Javascript object.
It starts with an open curly brace, and we are looking for a closing curly brace.
This can't be parsed with (proper) regular expressions because we need a way to count how many braces we've seen; we really need a stack of states.
We then have a state for being in a quote (e.g. {":}"} is a valid Javascript object; we don't count braced in the quote).
There's a state for being in an escape inside a quote, as in the previous example.
And finally there's a stack of states for tracking our current depth in braces.
In an imperative implementation we can track depth with an integer.
The terminal state is reached when we get back to a depth of 0.

```python
def get_js_object(text):
    depth = 0
    inquote = False
    escape = False
    for idx, char in enumerate(text):
        if escape:
            escape = False
            continue
        if char == '"':
            inquote = not inquote
        if char == '\\':
            escape = True
        if (not inquote) and char == '{':
            depth += 1
        if (not inquote) and char == '}':
            depth -= 1
            if depth <= 0:
                break
    return text[:idx+1]
```

Again this would be clearer if implemented with separate functions or objects for each state.
But even in an imperative solution keeping the state machine idea in mind makes it much easier to write a functioning implementation.