---
date: 2011-03-30 03:06:03+00:00
image: /images/latex.png
title: LaTeXing Multiple Equations
---

In mathematics and the (hard) sciences it’s important to be able to write documents with lots of equations, lots of figures and lots of references efficiently. This can be done in, for example, Microsoft Word, but the mathematics and theoretical physics community heavily prefer  $\TeX$  (and in particular  $\LaTeX$ ), so the bottom line is if you want to get papers published you’re going to have to get good at it.


There are a lot of resources for learning  $\LaTeX$  on the web, and a lot of people teach themselves from this (I know I did), but this can get you into some bad habits. For instance **eqnarray** gets the spacing around the equals signs all [wrong. ](http://www.tug.org/pracjourn/2006-4/madsen/madsen.pdf)(I typeset my thesis using exclusively eqnarray and didn’t notice this until it was pointed out to me). So a lot of people advocate **align** from AMSTeX, but align has it’s limitations too; it only comes with one alignment tab &. If you want to make a comment at the end of multiple equations (like “for  $x \in X$ “) or you want to have two equations and the second one breaks over two lines you can’t line the equations up properly; but there is a solution – IEEEeqnarray (which is an external class, IEEEtrantools, available from the IEEE). Stefan Moser has written an [excellent paper](http://moser.cm.nctu.edu.tw/docs/typeset_equations.pdf) covering everything I’ve said and much more, showing good ways to typeset equations.


<!--more-->


An interesting thing he points out is that  $\TeX$  sometimes typesets + as a binary operator, as in  ${} + a$  or as a unary (sign) operators as in  $+ a$  (note the extra spacing between the + and the a in the first example). The most common example of where + is typeset as a binary operator is when the thing following the + is a mathematical operator “\…” e.g. $+ \sin(x)$ as opposed to $+ {\sin(x)}$.


You can do some fiddling to trick  $\LaTeX$ ; indeed that’s how I produced the output above. Inserting an invisible character {} before the + sign makes  $\TeX$  treat + as a binary operator with more spacing preceding it. Encapsulating an operator in curly braces, like {\sin(x)}, means if + is the first thing on the line it is treated as a unary operator. But these fixed aren’t perfect in particular the invisible character trick isn’t good if we’re trying to break an equation over multiple lines (see Moser’s paper for more details); ${} + {\sin(x)}$ = ${} + \sin(x)$ is different to $+ \sin(x)$ – try it.


Some readers may think I’m being overly pedantic about a small space (1/2 of a quad), but  $\TeX$  advocates market it on its consistency in typesetting – these examples fail this and help you typeset inferior documents if you’re not careful.


There are many Integrated Development Environments for  $\TeX$  programs (like TeXworks) but the only one I know that lets you **preview equations** (it does a little compile of just the equations) is emacs with [auctex](http://www.gnu.org/software/auctex/preview-latex.html).


Incidentally there are other addons to  $\TeX$  around – e.g. [ConTeXt ](http://wiki.contextgarden.net/Main_Page)which appears to allow you to customise the layout of your document more easily than  $\LaTeX$  (which could be very useful in some situations; for instance if you want to wrap text around a figure), but at the cost of the author having to put more effort in laying out the document. As it happens, the standard way of producing multiple equations in ConTeXt seems to be as good as IEEEeqnalign. But while most scientific journals are still using  $\LaTeX$  so will most scientific authors.