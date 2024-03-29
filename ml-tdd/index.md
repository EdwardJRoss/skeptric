---
categories:
- testing
date: '2022-01-31T20:55:13+11:00'
image: /images/tdd.png
title: Test Driven Development in Machine Learning
---

Last weekend I read the first part of Kent Beck's [Test Driven Development: By Example](https://www.oreilly.com/library/view/test-driven-development/0321146530/).
He works through a simple example of programming with Test Driven Development in excruciating detail.
It shows how you can move in small steps when things get tricky, which is often better than a long debugging session.

I typically develop code iteratively in a Jupyter Notebook or IPython REPL.
I can pass the code I'm writing and interactively inspect it to make sure it works as inspected.
This is an efficient way to work, especially with unknown and messy data.
But you build up state in the REPL, and a model in your mind, which may diverge from the actual code.
And when the session is finished you won't know if you've introduced a regression, unless you also run tests.

I tried out Test Driven Development on a real problem, and found it very useful.
I found it often helped me notice coding issues much earlier, and resolve them much faster.
There were times when things got tricky and moving in small steps was very helpful.
At one point I introduced a regression and was able to notice and resolve it very quickly.
It didn't stop all issues at runtime, because existing code was untested, but I spent less time debugging.

The rest of this article goes through a detailed account of how I used TDD to implement this feature.

# The process

Kent Beck describes the TDD cycle as:

* Add a little test
* Run all the tests and fail
* Make a change
* Run the tests and succeed
* Refactor to remove duplication

I needed to have a little test that would check that the model had correctly loaded the weights.
A simple test would be to initialise the model with input weights, evaluate the embedding on an input, and then check that it matches the input weight.
However it wasn't immediately obvious to me how to even run the model.

So I started with a test that just ran the model on some data and checked the output dimensions.
I used some existing code to write and call the model, and after getting it wrong a few times the test passed.
Then I broke the test, changing the dimension check so it was wrong, to make sure the test was actually working.
When the test failed I undid my change and had an example of loading and calling a model in the test code.

Now I was in a position to write a test, and I had to work out where it would fit.

## Add a little test

The overall model contained different Embedding layers.
For some of the layers I needed to pass some pretrained weights.
There was a common function to construct the embeddings, which seemed like the right place to set the weights.

```python
import tensorflow as tf
from tf.keras import Sequential

def get_categorical_embedding(dim: int, vocab: list[str]) -> Sequential:
    return Sequential([
            tf.keras.layers.StringLookup(vocabulary=vocab, mask_token=None),
            # Add + 1 for out of vocabulary
            tf.keras.layers.Embedding(len(vocab) + 1, dim),
            ])
```

Having identified the function, I tried to think of a simple test.
The weights were stored in a file, which could easily be converted to a Pandas DataFrame with the categories on the index.
There are lots of other ways the data could be represented, but I thought I'd start a test with this and see how it looked.


```python
import pytest
import pandas as pd

@pytest.fixture
def weights():
    return pd.DataFrame([[0.5, 0.5], 
                         [1.0, 0.0]],
                         index=["cat", "dog"]
    )

def test_weighted_categorical_embedding_weights(weights):
    tokens = tf.convert_to_tensor(["cat", "dog"])
    model = get_categorical_embedding(
        dim=4, vocab=weights.index, weights=weights
    )
    embedding = model(tokens).numpy()

    expected = np.array(
        [[0.5, 0.5], [1.0, 0.0]]
    )
    assert np.allclose(embedding, expected)
```

However I realised I wasn't clear on what should happen to an out of vocabulary token.
The existing weights didn't have one, and I had to think of something to do.
I decided I should average the weights, and added it to the test.
This was a bit too much to handle all at once, but I forged on ahead.

## Run a test and fail

I ran the test and it failed.
First it failed because I made some mistakes in the test code.
When I fixed them it failed because `get_categorical_embedding` didn't have a weights argument.

## Make a Change

I added `weights` as an argument to the end of `get_categorical_embedding` and initialised it to `None`.
It wasn't immediately obvious what I needed to do next so I ran the tests again.

## Run a test and fail

The assertion now failed because the random initial weights didn't match the weights it was supposed to load.
This wasn't a surprise, since the code isn't loading the weights yet.
But now it's getting to the point where that's all I need to do.

It wasn't clear how to load the weights though.
I'd need to deal with the Out of Vocabulary tokens.
The right TDD thing to do here would be to add it to my TODO list, and put anything in that made the tests pass.
But instead I started thinking about how to get the weight matrix into the right shape.

## Write a Small Test

I wanted a function that took in my weights DataFrame and did something reasonable for the out of vocabulary tokens.
A reasonable thing to do would be to take the mean of all the other embeddings.
So I added a test to do that.

```python
def test_create_weight_matrix(weights):
    vocab = ["dog", "cat"]
    result = create_weight_matrix(weights, vocab)

    mean_embedding = [0.75, 0.25]

    expected = np.array([mean_embedding, [0.5, 0.5], [1.0, 0.0]])

    assert np.allclose(result, expected)
```

## Run a test and fail

This test now failed because the function didn't exist.
So I went ahead and wrote a simple implementation.

## Run a test and succeed

Then the test failed again; this time because of something like `0.7504 != 0.75`.
It seemed that Tensorflow slightly changed the representation.

I added the argument `rtol=1e-3` to `np.allclose` to check less precisely.
Then this test succeeded.
I updated other tests to use the same tolerance.

## Run a test and fail

Now I had the weights matrix with the out of vocabulary tokens I could add it to the embedding.

Looking at some examples it seemed like I should be able to add `weights=` to the [`Embedding`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Embedding) layer.
It wasn't explicitly documented, but it passes some keyword argument so I thought I would try it.
However the test still failed, the initialisation was still random.

## Run a test and fail

It looked like there were two ways I could do it; manually call [`set_weights`](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Layer#set_weights), which first requires instantiating the layer, or passing a [`Constant`](https://www.tensorflow.org/api_docs/python/tf/keras/initializers/Constant) Initializer, which claims it can only take scalar values.
After seeing an example with Constant online I tried it and this test passed, but my first test calling the model failed.

## Run a test and succeed

My code didn't handle the case where the weights were `None`; I had introduced a regression.
It was simple to write an `if` statement and this test passed.

## Refactor

Now all my tests were passing I could take some time to refactor.
I cleaned up some test code, but the model code looked good so I continued.

## Write a small test

The full model now needed to get the embeddings with the weights.
I wrote a simple test calling the model with an expected output.

## Run a test and fail

The test immediately failed because the model didn't have arguments for weights.
I added them and propagated them to the call to `get_categorical_embedding`.

There were a cascade of small errors with how I wrote this, which I quickly fixed.

## Run a test and succeed

Then I got a strange failure about dimensions not matching.
It wasn't obvious to me what was happening here.
The dimensions it mentioned were 2 (which was the dimension of the weights) and 4.
It then dawned on me that the embedding layer in my test had dimension 4, and the weights had dimension 2.
I fixed the test and added a note to check the dimensions matched so this failure occurred at instantiation.

The tests were all passing now, and it seemed like it was working.

## Checking the dimensions

I wrote a test to check if the embedding and weight dimensions don't match it should raise a `ValueError`.
The test failed as expected.

I then wrote the assertion and the test passed.

To make sure the error message was right, since sometimes I make mistakes with this, I changed the test to check for a `RuntimeError`.
The test failed with the right message, and so I reverted my change.

However I'm not really checking my error message, and it is open to regression; a good extension would be to check the message itself.

## Running the whole thing

I was now confident enough to run the whole process, which takes about ten minutes.
It *still* failed a couple of times.

First I made a simple typo in a variable name.
I ran `pyflakes` and noticed it would pick it up.
I configured `pyflakes` to ignore all the other existing issues and added it to my `make test` command.

Next it failed to save the model at the end.
I had passed a Pandas series as the vocabulary, and Tensorflow didn't know how to serialise it.
Converting to a list or numpy array fixed the problem.

After that it worked perfectly.
All the testing gave me more confidence it was doing what I expected, rather than just running with the wrong weights.
I'm pretty happy with my experience of TDD and want to keep trying it.