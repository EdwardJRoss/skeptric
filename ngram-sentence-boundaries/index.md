---
categories:
- nlp
date: '2021-01-20T19:04:33+11:00'
image: /images/ngram-sentence-probabilities.png
title: Sentence Boundaries in N-gram Language Models
---

An N-gram language model guesses the next possible word by looking at how frequently is has previously occurred after the previous N-1 words.
I think this is how my mobile phone suggests completions of text; if I type "I am" it suggests "glad", "not" or "very" which are likely occurrences.
To make everything add up you have to have special markers for the start and end of the sentence, and the I think the best way is to make them *the same marker*.

I'm reading through the December 30, 2020 draft of [Speech and Language Processing, by Jurafsky and Martin](https://web.stanford.edu/~jurafsky/slp3/), in particular Chapter 3 *N-gram Language Models*.
To create the language model you simply count how often each word follows the previous N-1 words to create a count matrix, and then normalise each row to 1 so it's a conditional probability.
Then, under the assumption that the probability of a word just depends on the previous N-1 words, you can calculate the probability of every sentence.
As shown in Exercise 3.5 of the text if you don't have an end sentence marker then the probabilities over every possible sentence don't add up to 1.

As an example consider the corpus of two sentences, with the special markers to start a sentence, `<s>`, and end a sentence `</s>`:

```
<s> I am Sam </s>
<s> Sam I am </s>
```

Then we can generate the counts for a bigram model on words (i.e. N=2), with the first word on the left

| **Counts**           | &lt;s&gt; | I | am | Sam | &lt;/s&gt; |
|------------|-----------|---|----|-----|------------|
| &lt;s&gt;  | 0         | 1 | 0  | 1   | 0          |
| I          | 0         | 0 | 2  | 0   | 0          |
| am         | 0         | 0 | 0  | 1   | 1          |
| Sam        | 0         | 1 | 1  | 0   | 0          |
| &lt;/s&gt; | 0         | 0 | 0  | 0   | 0          |

Notice that structurally the first column and last row *have* to be zero; a start sentence marker can never occur after any token and no token can follow the end sentence marker.
Now if we wanted to use add-k smoothing to allow for combinations of words not in the text, then we can't add k to the first column or last row, but we want to add it to every other cell.
I suggest instead unifying `<s>` and `</s>` as the same token, giving the count matrix:

| **Counts** | &lt;s&gt; | I | am | Sam |
|------------|-----------|---|----|-----|
| &lt;s&gt;  | 0         | 1 | 0  | 1   |
| I          | 0         | 0 | 2  | 0   |
| am         | 1         | 0 | 0  | 1   |
| Sam        | 0         | 1 | 1  | 0   |

In this matrix any cell can be non-zero, as long as we allow empty sentences `<s> <s>`.
In this case we can convert them to conditional probabilities by dividing each row by its sum:


| **P(column \| row)** | &lt;s&gt; | I   | am  | Sam |
|--------------------|-----------|-----|-----|-----|
| &lt;s&gt;          | 0         | 0.5 | 0   | 0.5 |
| I                  | 0         | 0   | 1   | 0   |
| am                 | 0.5       | 0   | 0   | 0.5 |
| Sam                | 0         | 0.5 | 0.5 | 0   |

We can use this to calculate the probability of any sentence under the bigram model, for example the sentence `<s> Sam I am <s>` has probability (using `.` in place of `<s>`)

$$P(\rm{Sam\ I\ am}) = P(\rm{Sam} \vert .) P(\rm{I} \vert \rm{Sam}) P(\rm{am} \vert \rm{I}) P(. \vert \rm{am})= 0.5 \times 0.5 \times 1 \times 0.5 = 0.125$$

Note this generalises and we can get sentences like `<s> I am <s>` or `<s> Sam I am Sam am <s>`, but we can't possibly get `<s> am I Sam <s>` since $P(\rm{am}| .) = 0$.

The add-k method adds a constant k to every element of the matrix.
As discussed we can only do this naively if we've got a single sentence boundary token.

| **Add 0.5 counts** | &lt;s&gt; | I   | am  | Sam |
|--------------------|-----------|-----|-----|-----|
| &lt;s&gt;          | 0.5       | 1.5 | 0.5 | 1.5 |
| I                  | 0.5       | 0.5 | 2.5 | 0.5 |
| am                 | 1.5       | 0.5 | 0.5 | 1.5 |
| Sam                | 0.5       | 1.5 | 1.5 | 0.5 |

We can then normalise these into probabilities:

| **P*(column \| row)** | &lt;s&gt; | I     | am    | Sam   |
|-----------------------|-----------|-------|-------|-------|
| &lt;s&gt;             | 0.125     | 0.375 | 0.125 | 0.375 |
| I                     | 0.125     | 0.125 | 0.625 | 0.125 |
| am                    | 0.375     | 0.125 | 0.125 | 0.375 |
| Sam                   | 0.125     | 0.375 | 0.375 | 0.125 |

Now any combination of the words is possible, even the empty sentence `<s> <s>` (with probability 12.5%).

If we move to a higher order model such as a trigram model we need to add more padding tokens *at the start of the sentence*, which again should all be the same.
For example with our corpus the counts would look like:

```
<s> <s> I am Sam <s>
<s> <s> Sam I am <s>
```

| **Counts**          | &lt;s&gt; | I | am | Sam |
|---------------------|-----------|---|----|-----|
| &lt;s&gt; &lt;s&gt; | 0         | 1 | 0  | 1   |
| &lt;s&gt; I         | 0         | 0 | 1  | 0   |
| &lt;s&gt; Sam       | 0         | 1 | 0  | 0   |
| I am                | 1         | 0 | 0  | 1   |
| am Sam              | 1         | 0 | 0  | 0   |
| Sam I               | 0         | 0 | 1  | 0   |

Note that we don't want an additional padding token at the end, because given any token followed by `<s>` the next token must be constrained to be `<s>`.

This is one of those little tricks in how you frame a problem that makes the calculations much easier in practice.