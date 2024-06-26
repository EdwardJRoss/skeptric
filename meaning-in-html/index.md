---
categories:
- html
- nlp
date: '2022-06-18T11:25:40+10:00'
image: /images/html_equivalent.png
title: Finding Meaning in HTML
---

HTML is one of the most common forms of communication today.
Emails, wikis, blogs, and many forums ultimately use some sort of HTML to communicate things such as emphasis, cross-links and structure.
However in Natural Language Processing on HTML the most common practice is to throw away all the markup.
A lot of markup is meaningless, from a communication perspective, but removing all of it is throwing out the baby with the bathwater, and there are better ways to do this.

I've written before about [why you should use HTML in NLP](/html-nlp) and how we can extract the text with context by [source mapping](/source-mapping-html).
In this article I want to delve into what semantics we want to extract from HTML and outline some steps to do that.
Ultimately this is for practical reasons; I want to extract information and meaning from webpages, and some of that meaning is encoded in the HTML.

# Linguistics of HTML

HTML is a framework for a wide variety of things from describing the scaffolding of how a page should be visually laid out, to a data structure that can be dynamically modified with Javascript for applications, through to a communication medium.
I'm going to focus on the latter, thinking of using HTML in the same way we use spoken languages.

One useful feature of HTML is span elements like `<em>` and `<strong>` (and their less semantic cousins `<i>`, `<b>` and `<u>`) that signify that this piece of information is particularly important.
These are analogous to when we are speaking we change the way say a word to bring attention to it; as can be seen in the corresponding emphasis element in the [Speech Synthesis Markup Language](https://www.w3.org/TR/speech-synthesis11/#S3.2.2).
This has a long history in typesetting where different fonts are used to convey important parts of information (or sometimes different sorts of information such as names of works of art).
It's purely a historical accident that we consider quotes, indicating that someone is speaking, to be part of written language but not emphasis (but long quotes of other source are represented typographically and in HTML are `<blockquote>`).

HTML can also refer to other things such as images, audio, or video embedded in the HTML or a hyperlink referencing another web page.
This is the written equivalent of "pointing" at something as you talk about it, demonstrating what you're talking about (although in some cases the media *is* the only subject, which is less relevant here).
These have written analogues in pictures, and cross-references (through footnotes or bibliographies).

HTML can also indicate structure which is more implicit in communication.
A simple structure is the paragraph, `<p>`, which indicates a change in topic, and has a long tradition in written work.
In fact it used to be explicitly denoted with a [pilcrow](https://en.wikipedia.org/wiki/Pilcrow) ¶, and had language evolved differently we may not need this.
In spoken language the change of topic is often indicated by a short break or a change in pace or body language.
HTML also has more complicated structures like hierarchical headings from `<h1>` to `<h6>` which are something of a generalisation of epics consisting of stories consisting of chapters, which are useful for complex communication.
There's also lists, `<ul>` and `<ol>` which in spoken language we may enumerate on our fingers as we change pace.

The exact way HTML is used to convey these ideas depends on the context.
Due to cultural differences or limitations of a particular environment the same concept can be represented in different ways.
Consider the following HTML adapted from a job advertisement:

```html
<p><strong><em>The Client</em></strong></p>
<p>Our client is ... </p>
<p><em><strong>The Job</strong></em><br />
This is a job for a talented person ... </p>
<p>You'll be responsible for ... <br />
• Managing ... <br />
• Reporting ... <br />
</p>
```

On the first line they're using strong and emphasis to create a header.
On the third line they do the same thing but the order of emphasis and strong are switched, then they use break to end the heading rather than create a new paragraph.
Finally a list is creates using bullet points.
A more semantic way to communicate the same thing would be:

```html
<section>
    <h1>The Client</h1>
    <p>Our client is ... </p>
</section>
<section>
    <h1>The Job</h1>
    <p>This is a job for a talented person ... </p>
    <p>You'll be responsible for ... </p>
    <ul>
        <li>Managing ... </li>
        <li>Reporting ... </li>
    </ul>
</section>
```

How do we extract the key communication concepts in HTML?

# Algebra of Semantic HTML

There are often different representations of HTML that are equivalent in communication.
In the previous example `<strong><em>` has the same effect as `<em><strong>`; the elements commute.
They're also idempotent; `<em><em>` has the same effect as `<em>`.
They're not exactly equivalent; it's easy to write CSS that makes them different, but in practice they tend to mean the same thing.

The rules of HTML prevent nesting paragraphs; by the [whatwg standard](https://html.spec.whatwg.org/multipage/grouping-content.html#the-p-element) the end of a paragraph is inferred by another paragraph, a header and many other things.
That is `<p><p></p>` is interpreted as `<p></p><p></p>`.

Sectioning is implicit; a header element like `<h3>` implicitly starts a subsection if preceded by a higher level header (like `<h2>`), a sibling section if preceded by an equal level header (another `<h3>`), and a new section if preceded by a lower level header (like `<h4>`).
This can be explicitly called out with the `<section>` tags, but rarely seems to in practice.
This gives an outline-like tree structure to HTML documents (in contrast to the DOM tree like structure).

# Markup Language Processing

We can think of these meaningful markup as features for a model.
A heading for a section may be represented with a `<h1>` or with a paragraph consisting of a single sentence in a `<em><strong>`.
We can extract these and feed them in an appropriate way to a model to detect headers.
We can consider a HTML document not as a tree but as a series of sections containing a header and paragraphs (where a paragraph includes things like a list).
The exact choices depend on the usecase.