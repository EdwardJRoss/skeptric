---
date: '2021-07-12T15:12:29+10:00'
image: /images/spelling-fixes.png
title: Spellchecking Articles with Aspell
---

I write enough text in Emacs that it's worth using a spellchecker.
As I started comparing options I ended up spellchecking all my existing articles on this website.
While I didn't get everything, I fixed 475 typos across 222 articles in a few hours, and built up a custom dictionary of my most common words.

The Emacs manual on [spelling](https://www.gnu.org/software/emacs/manual/html_node/emacs/Spelling.html) says it supports Hunspell, Aspell, Ispell and Enchant.
For English the choice is between Hunspell and Aspell; Ispell is a predecessor or Aspell, and Enchant wraps other spelling libraries (most notably specific ones for Turkish and Finnish).
Running some examples I found Aspell gave much better suggestions so I decided to go with that.

For spell checking existing posts I found [a guide from ThorneLabs](https://thornelabs.net/posts/spell-checking-many-posts-with-aspell-and-a-custom-dictionary.html) to output all the words Aspell marks as misspelled, which I adapted a little.

```sh
for POST in content/post/*
do
    cat $POST | aspell list --ignore 2 --mode markdown --camel-case -l en_AU
done | sort | uniq -c | sort -nr | sed -E 's/^ *[0-9]+ *//' > words.txt
```

The flags are:

* `--ignore 2` ignore any words of 2 or less characters
* `--camel-case` accept CamelCased words as valid spellings (occurs a lot in some articles like [job posting schemata](/schema-jobposting))
* `-l en_AU` set the language to Australian English
* `--mode markdown` treat the files as markdown

The pipes at the end are a [shell transformation](/shell-etl) to sort the words Aspell by frequency, and output it to a file called `words.txt`.
I could then go through this file and sort out valid words not in Aspell's dictionary from actually misspelled words.
Sometimes I would need to check the context using `grep -r '\bmodularity\b' content/post`, or check Aspell's suggestions with `echo modularity | aspell -a`.

I ended up with a list of words that I use frequently that needed to be added to my personal dictionary at `$HOME/.aspell.en.pws`.
These were a mixture of technical words (integrable, geocoder, quantile, submodule), technologies (Jupyter, spaCy, Javascript), acronyms (WSL, NLP, RDF), contractions (codebase, whitespace, substring, postamble) and TeX commands (frac, ldots, infty).
Unfortunately I had to remove any phrases (such as *ad hoc*), words with hyphens (such as Gell-Mann), or unicode characters (such as Dieudonné or Möbius) from the dictionary or [Aspell would complain](https://web.archive.org/web/20190927234225/http://aspell.net/man-html/Words-With-Symbols-in-Them.html).

I ended up with a second list of words that I'd misspelled frequently, and could run back through aspell to correct.
Apparently I commonly misspelled words like hierarchy, manageable, metres, occurred, focusing, ridiculous, and inheritance.
Another thing I get wrong is capitalisation of name like GitHub, TensorFlow, or Emacs.
I fixed the most common ones and ones that were obviously wrong with sed, then viewed the diff in Magit (using [`magit-diff-refine-hunk`](https://magit.vc/manual/magit/Diff-Options.html) to highlight the changed words as in the cover image).
There were a couple cases I needed to remove, like changes in URLs or quotes, but the majority changes were correct.

I then could add the dictionary to my [dotfiles](/portable-custom-config), and use it in Emacs through the `ispell` commands or `flyspell` commands (which I still need to configure).
I'm not the only person to have this idea and it's interesting to see [other people's personal dictionaries](https://github.com/search?o=desc&q=filename%3Aaspell.en.pws&s=&type=Code).
