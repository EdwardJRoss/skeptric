---
date: 2013-08-14 04:45:28+00:00
image: /images/dvi_xxd.png
title: DVI by example
---

The Device Independent File Format (DVI) is the output format of Knuth’s TeX82; modern TeX engines (pdfTeX, luaTeX) output straight to Adobe’s Portable document format (PDF). However TeX82 and DVI still work as well today as they did when they were written; DVI files are easily cast to postscript or PDF.


The defining reference for DVI files is David R Fuch’s [article](http://www.tug.org/TUGboat/Articles/tb03-2/tb06software.pdf) in TUGboat Vol 3 No 2.


To find out what information is contained in a particular DVI file use Knuth’s [dvitype](http://www.ctan.org/pkg/dvitype), which outputs the operations contained in the bytecode in human readable format.


This article goes into gory detail the instructions contained in a very simple DVI file.


<!--more-->


Overview of the format
----------------------


DVI is designed as a description of how to typeset horizontally and vertically a black and white document (in a left-to-right alphabetic language) for printing.


The basic operations are to typeset a character (in the specified font) optionally advancing by the character’s width, to typeset a rectangular box optionally advancing the box’s width, and setting variables (including the font and position where to typeset).


The file is encoded as a series of 8-bit bytes; the first byte is an *operation* followed by a given number of arguments. Each argument is either a fixed number of bytes in length, or has a length given by a prior argument. There are four kinds of parameter: unsigned integer (represented by its bytes as a binary number), signed integer (using [two’s complement](http://en.wikipedia.org/wiki/Two%27s_complement)), pointer, or strings (represented as a series of 1-byte character codes) used for information and filesystems.


Internally there are a number of state parameters; the current font f (a 4-byte signed integer), the position and spacing variables (h, v, w, x, y, z) (each a 4-byte signed integer) and a stack of position and spacing variables. (h, v) represents the point h units to the right and v units down from the top left corner of the page. w, x are horizontal spacing parameters and y, z are vertical spacing parameters. The units are determined in the file itself.


Below are the operations, I will use the notation n[4] to represent a parameter of 4 bytes, and curly braces {} to represent a range of commands.I will use characters a, b, c, … to represent signed integers; i, j, k, l, m, n, … to represent unsigned integers; p, q, … to represent pointers and A, B, … to represent characters and X to represent a custom (user implemented) type. These types have been inferred and not checked, so use with caution.


<table>
<tbody>
<tr>
<th>Hex code</th>
<th>Name</th>
<th>Params</th>
<th>Function</th>
</tr>
<tr>
<td>{0-7F}</td>
<td>set_char_{1-127}</td>
<td></td>
<td>Typeset character {1-127} in font f at (h, v), then advance h by the width of that character</td>
</tr>
<tr>
<td>{80-83}</td>
<td>set{1-4}</td>
<td>m[{1-4}]</td>
<td>Typeset character m in font f at (h, v), then advance h by the width of that character</td>
</tr>
<tr>
<td>84</td>
<td>set_rule</td>
<td>a[4], b[4]</td>
<td>Typeset box of width a, height b at (h, v), then advance h by a</td>
</tr>
<tr>
<td>{85-88}</td>
<td>put{1-4}</td>
<td>m[{1-4}]</td>
<td>Typeset character m in font f</td>
</tr>
<tr>
<td>89</td>
<td>put_rule</td>
<td>a[4], b[4]</td>
<td>Typeset box of width a, height b at (h, v)</td>
</tr>
<tr>
<td>8A</td>
<td>nop</td>
<td></td>
<td>No operation</td>
</tr>
<tr>
<td>8B</td>
<td>bop</td>
<td>a{0-9}[4] p[4]</td>
<td>New page, a{0-9} are TeX registers `\count{0-9}` to identify the page for reference, p is a pointer to previous page (or -1 for first page). All state is reset</td>
</tr>
<tr>
<td>8C</td>
<td>eop</td>
<td></td>
<td>End of page, output page. Stack should be empty</td>
</tr>
<tr>
<td>8D</td>
<td>push</td>
<td></td>
<td>Push (h, v, w, x, y, z) onto stack</td>
</tr>
<tr>
<td>8E</td>
<td>pop</td>
<td></td>
<td>Pop from stack, and set variables</td>
</tr>
<tr>
<td>{8F-92}</td>
<td>right{1-4}</td>
<td>a[{1-4}]</td>
<td>Advance h by a</td>
</tr>
<tr>
<td>93</td>
<td>w0</td>
<td></td>
<td>Advance h by w</td>
</tr>
<tr>
<td>{94-97}</td>
<td>w{1-4}</td>
<td>a[{1-4}]</td>
<td>Set w to a and advance h by w</td>
</tr>
<tr>
<td>98</td>
<td>x0</td>
<td></td>
<td>Advance h by x</td>
</tr>
<tr>
<td>{99-9C}</td>
<td>x{1-4}</td>
<td>a[{1-4}]</td>
<td>Set x to a and advance h by x</td>
</tr>
<tr>
<td>{9D-A0}</td>
<td>down{1-4}</td>
<td>a[{1-4}]</td>
<td>Advance v by a</td>
</tr>
<tr>
<td>A1</td>
<td>y0</td>
<td></td>
<td>Advance v by y</td>
</tr>
<tr>
<td>{A2-A5}</td>
<td>y{1-4}</td>
<td>a[{1-4}]</td>
<td>Set y to a and advance v by y</td>
</tr>
<tr>
<td>A6</td>
<td>z0</td>
<td></td>
<td>Advance v by z</td>
</tr>
<tr>
<td>{A7-AA}</td>
<td>z{1-4}</td>
<td>a[{1-4}]</td>
<td>Set z to a and advance v by z</td>
</tr>
<tr>
<td>{AB-EA}</td>
<td>fnt_num_{0-64}</td>
<td></td>
<td>Set f to {0-64}</td>
</tr>
<tr>
<td>{EB-ED}</td>
<td>fnt{1-3}</td>
<td>m[{1-3}]</td>
<td>Set f to m</td>
</tr>
<tr>
<td>EE</td>
<td>fnt4</td>
<td>a[4]</td>
<td>Set f to a</td>
</tr>
<tr>
<td>{EF-F2}</td>
<td>xxx{1-4}</td>
<td>m[1-4] X[m]</td>
<td>Implementation dependent; nop in general. Sent via TeX’s `\special`.</td>
</tr>
<tr>
<td>{F3-F5}</td>
<td>fnt_def{1-3}</td>
<td>i[{1-3}] j[4] k[4] l[4] m[1] n[1] A[m+n]</td>
<td>Sets font h to be the font loaded from subpath “A[0:m]/A[m:n]” of the standard fonts directory, with checksum j, scaled by k/l. k and l must be less than 2^27</td>
</tr>
<tr>
<td>F6</td>
<td>fnt_def4</td>
<td>a[4] j[4] k[4] l[4] m[1] n[1] A[m+n]</td>
<td>Sets font a as for F3-F5.</td>
</tr>
<tr>
<td>F7</td>
<td>pre</td>
<td>i[1] j[4] k[4] l[4] m[1] A[m]</td>
<td>Preamble; i is DVI version number which is 2. l is considered a magnification; 1 unit is set to $\frac{j}{k} 10^{-7} \mbox{m}$ and the entire document is scaled by a factor of $\frac{l}{1000}$. A is an information header</td>
</tr>
<tr>
<td>F8</td>
<td>post</td>
<td>See below</td>
<td>Postamble; see below</td>
</tr>
<tr>
<td>F9</td>
<td>post_post</td>
<td>See below</td>
<td>Post postamble; see below</td>
</tr>
<tr>
<td>FA-FF</td>
<td>undefined</td>
<td></td>
<td></td>
</tr>
</tbody>
</table>
A dvi must start with a preamble, followed by 1 or more pages ends with a postamble. A page is a bop followed by any instructions and terminating in an eop. The only operations that can go between these chunks are nop and font definitions.


The postamble has a 4-byte pointer to the beginning of the last page, then the parameters j, k, and l from the preamble (called numerator, denominator, and magnification respectively), a 4-byte signed integer giving the height+depth of the tallest page, a 4-byte signed integer giving the width or the widest page, a 2-byte unsigned integer giving the maximum stack depth in the DVI, and a 2-byte unsigned integer giving the total number of pages (bop commands).


Then each font must be defined. Each font must be defined exactly twice in the document, once before its first use (before the postamble) and once in the postamble.


The postamble concludes with the post-postamble, which contains a 4-byte pointer to the beginning of the postamble, followed by the version number i from the preamble followed by 4 of more of DFs (why not 8As? I have no idea).


The file is thus designed to be read forwards (one operation at a time) or backwards (one page at a time), and useful set-up information (page size and maximum stack depth) are at the back. The files are typically very compact, and because of their linear nature can be processed rapidly. Notice that the page is the minimum displayable unit; a page may be typeset non-linearly, but after a bop the page can no longer be affected.


Simple example
--------------


I typeset the following file Hello.tex



    Hello World!
    \bye



and then ran commands



    tex Hello.tex
    xxd Hello.dvi



which yielded



    0000000: f702 0183 92c0 1c3b 0000 0000 03e8 1b20  .......;.......
    0000010: 5465 5820 6f75 7470 7574 2032 3031 332e  TeX output 2013.
    0000020: 3038 2e31 323a 3138 3034 8b00 0000 0100  08.12:1804......
    0000030: 0000 0000 0000 0000 0000 0000 0000 0000  ................
    0000040: 0000 0000 0000 0000 0000 0000 0000 0000  ................
    0000050: 0000 00ff ffff ff8d 9ff2 0000 8ea0 0283  ................
    0000060: 33da 8da0 fd86 cc26 8d91 1400 00f3 004b  3......&.......K
    0000070: f160 7900 0a00 0000 0a00 0000 0563 6d72  .`y..........cmr
    0000080: 3130 ab48 656c 6c6f 9103 5555 5791 ff2a  10.Hello..UUW..*
    0000090: aa6f 726c 6421 8e8e 9f18 0000 8d92 00e8  .orld!..........
    00000a0: 60a3 318e 8cf8 0000 002a 0183 92c0 1c3b  `.1......*.....;
    00000b0: 0000 0000 03e8 029b 33da 01d5 c147 0002  ........3....G..
    00000c0: 0001 f300 4bf1 6079 000a 0000 000a 0000  ....K.`y........
    00000d0: 0005 636d 7231 30f9 0000 00a5 02df dfdf  ..cmr10.........
    00000e0: dfdf dfdf                                ....

Let’s walk through this byte by byte.


### Preamble

    f7 02 018392 c01c3b0000 000003e8
       1b 20 54 65 58 20 6f 75 74 70 75 74 20 32 30 31 33 2e 30 38 2e 31 32 3a 31 38 30 34

The first line starts with the pre opcode, followed by the version 02, numerator = 25400000, denominator = 473628672 and magnitude = 1000. This means 1 unit is  ${25400000 \over 473628672} 10^{-7}\mbox{m} = {0.0254 \over 72.27 * 2^{16}} \mbox{m}$ . There are 72.27 [standard points](http://en.wikipedia.org/wiki/Point_%28typography%29#Traditional_American_point_system) in an inch and 2.54 cm in an inch, so a standard point is  ${72.27 \over 0.0254} \mbox{m} = 2^{16} \mbox{unit}$ . Thus a unit is  $2^{-16}$  standard points; what TeX calls a scaled point (sp). The document is then scaled by 1000/1000=1.


The second line is the documentation string; 1b states it consists of 27 bytes. The bytes then form the ASCII string


     TeX output 2013.08.12:1804
Evidently I ran TeX at 18:04 pm on the 12th of August 2013 A.D.


### Beginning of page

    8b 00000001 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 00000000 ffffff

The preoperation is followed by the values in the registers count 0 through 9. By default TeX uses `\count0` for the page number and doesn’t affect the other counts, so we get `\count0 = 1` (first page), `\count{1-9}=0`. Finally since this is the first page the pointer to the previous page is -1 (ffffff).


### Setup

    8d
    9f f20000
    8e

Pushes the current (h, v, w, x, y, z) onto the stack; (0, 0, 0, 0, 0), then moves v down by -917504 sp, then pops the stack resetting v; overall achieving nothing. I presume this would probably put a header in a more complex document.



    a0 028333da
    8d
    a0 fd86cc26
    8d

Move v down by 42152922 sp = 8.84 inches, pushes onto the stack (0, 42152922, 0, 0, 0, 0), then drops v down -41497562 sp, then pushes onto the stack and pushes onto the stack (0, 655360, 0, 0, 0, 0) (this is 10 standard points down from the top of the page).



    91 140000

Moves h right by 1310720 sp = 20 standard points.


### Define font

    f3 00 4bf16079 000a0000 000a0000
       00 05 63 6d 72 31 30

f3 is fnt_def1, and we set font number 0, the checksum is 4bf16079 and we the “scale size” and “design size” are both 655360 sp = 10 standard points, so it comes out at its default size of 10 points.
The next line is the path reference: the length of the directory name is 0, the length of the file name is 5, and the file name is     cmr10.


### Hello world!

    ab

Set the font to font 0 (cmr10)



    48 65 6c 6c 6f

Typeset     Hello, advancing at each character.



    91 035555
    57

Move right 3 1/3 standard points and typeset     w, advancing.



    91 ff2aaa
    6f 72 6c 64 21

We now move left by 5/6ths of a standard point (this is TeX performing kerning) and then typeset     orld!


### Page number

    8e
    8e
    9f 180000
    8d

Pop twice, giving the current state (0, 42152922, 0, 0, 0, 0), so we are down about 8.84 inches; then move down another 1572864 sp = 24 standard points and push this position onto the stack.



    92 00e860a3
    31
    8e
    8c

Move right 15229091 sp ~= 3.2 inches which is almost half way accross the 6.5 inch width (at the end), typeset     1 then pop the stack and end the page.


### Postamble

    f8 0000002a
       018392c0 1c3b0000 000003e8
       029b33da 01d5c147
       0002
       0001

Begin the postamble; the last page begins at the 42nd byte (2a).
Restate the numerator, denominator and magnification.
The maximum page height+depth is 43725788 sp ~= 9.23 inches, and the maximum page width is 30785863 sp ~= 6.5 inches.
The maximum stack depth is 2.
There is 1 page.



    f3 00 4bf16079 000a0000 000a0000
       00 05 63 6d 72 31 30

Repeat the font definition.


### Post-post amble

    f9 000000a5 02

The post-postamble declares the post-amble starts at byte 165, and reiterates this is version 02 of DVI.



    df df df df df df df

Finally pad with df’s; the file should be a multiple of 4 bytes.


Limitations and flexibilities
-----------------------------


Keep in mind the DVI format was invented in 1979; it’s amazing how well it’s stood up. Most parameters can be specified by a 32-bit number; this means distances are specified by more than 1 part in 2 billion, and numbers by more than 1 part in 4 billion.


In particular using the scaled point we can specify positions to half a nanometre for pages up to 23 metres in length! (And we could magnify this by more than 16 orders of magnitude!)


Our fonts can have up to 2^32 characters; so it can encode even fonts that conform to the widest Unicode convention today UTF-32. We can also have up to 2^32 fonts; we could use this to hold fonts in different colours, styles (e.g. bold, italic, …) or perhaps something more exotic like angled lines of different slopes and thicknesses.


However we can’t **embed** our fonts, if someone wants to view the DVI they have to ensure they have all the correct fonts, in the correct directory (which is determined in part by the DVI viewer) which can be no more than 1 level deep. Also there is no way in the DVI specification of fetching information about the font (height, width, etc.), so if we want to put an accent on a character we have to hard code its properties into the DVI. This would be OK if put_char didn’t advance by the character’s width; this badly breaks orthogonality of the system. Another bad example would be right-to-left text; you would have to hard code jumps relating to each character’s width.


Because we have to have a pointer to the postamble the DVI can’t much more than 4Gb in size; even if all our characters are Unicode, and we have 80 characters per line and 50 lines per page we should still be able to produce a document well over a hundred thousand pages in size. The stack can be at most 65535 level deep (which I think you’d only exceed if you were trying). The 10 counters also allow us to collate our document in a highly non-linear fashion.


It’s odd there are two widths w, x and two heights y and z; in fact zero would suffice by directly adding to h and v. I think the reason for these variables must be buried in TeX.


There’s no way to insert images, change the colour of text, and for computer viewers no way to insert hyperlinks, videos and sounds or take user input. There’s also no way to draw complex objects without either rendering them specially as fonts, or explicitly constructing them using rectangular boxes as pixels. However the instruction xxx (from TeX’s `\special{}`) allows us to implement these things on an ad hoc basis. So it is extremely extensible; but we have to work to make those extensions portable.
